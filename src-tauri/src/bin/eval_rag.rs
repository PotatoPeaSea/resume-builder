//! RAG retrieval-quality eval harness.
//!
//! Seeds a throwaway SQLite DB with a fixed bullet corpus, runs the *real*
//! embedding + KNN retrieval pipeline for each job-description query, then uses a
//! local Ollama model as an LLM judge to grade the relevance of each retrieved
//! bullet. Computes ranking metrics (Precision@k, nDCG@k, MRR, mean grade, and —
//! where ground truth is provided — Recall@k) and writes a timestamped report.
//!
//! Usage:
//!   cargo run --bin eval_rag -- [--dataset PATH] [--top-k N] [--model TAG]
//!                               [--ollama-url URL] [--out DIR] [--judge-all-pairs]
//!
//! Requires a running Ollama server with the judge model pulled
//! (`ollama pull <model>`).

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use resume_builder_lib::db::init_db;
use resume_builder_lib::llm::{generate_ollama_opts, LlmSettings};
use resume_builder_lib::rag::commands::{search_similar_conn, ScoredBullet};
use resume_builder_lib::rag::EmbeddingModel;

// ─── Dataset schema ───

#[derive(Debug, Deserialize)]
struct Dataset {
    corpus: Vec<CorpusBullet>,
    queries: Vec<Query>,
}

#[derive(Debug, Deserialize)]
struct CorpusBullet {
    key: String,
    #[serde(default)]
    experience: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct Query {
    id: String,
    job_description: String,
    /// Optional ground truth: corpus keys that *should* be retrieved.
    #[serde(default)]
    relevant_keys: Vec<String>,
}

// ─── Config ───

struct Config {
    dataset: PathBuf,
    top_k: i32,
    model: String,
    ollama_url: String,
    out_dir: PathBuf,
    judge_all_pairs: bool,
}

fn parse_args() -> Config {
    let defaults = LlmSettings::default();
    let mut cfg = Config {
        dataset: PathBuf::from("evals/rag/dataset.json"),
        top_k: 10,
        model: defaults.ollama_model.unwrap_or_else(|| "qwen3.5:9b".to_string()),
        ollama_url: defaults
            .ollama_url
            .unwrap_or_else(|| "http://localhost:11434".to_string()),
        out_dir: PathBuf::from("evals/rag/results"),
        judge_all_pairs: false,
    };

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dataset" => cfg.dataset = PathBuf::from(args.next().expect("--dataset needs a value")),
            "--top-k" => {
                cfg.top_k = args
                    .next()
                    .expect("--top-k needs a value")
                    .parse()
                    .expect("--top-k must be an integer")
            }
            "--model" => cfg.model = args.next().expect("--model needs a value"),
            "--ollama-url" => cfg.ollama_url = args.next().expect("--ollama-url needs a value"),
            "--out" => cfg.out_dir = PathBuf::from(args.next().expect("--out needs a value")),
            "--judge-all-pairs" => cfg.judge_all_pairs = true,
            "-h" | "--help" => {
                println!(
                    "Usage: eval_rag [--dataset PATH] [--top-k N] [--model TAG] \
                     [--ollama-url URL] [--out DIR] [--judge-all-pairs]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(2);
            }
        }
    }
    cfg
}

// ─── Report schema ───

#[derive(Debug, Serialize)]
struct JudgedBullet {
    key: String,
    rank: usize,
    distance: f32,
    grade: u8,
    content: String,
}

#[derive(Debug, Serialize)]
struct QueryResult {
    id: String,
    retrieved: usize,
    precision_at_k: f64,
    ndcg_at_k: f64,
    mrr: f64,
    mean_grade: f64,
    /// Recall against `relevant_keys` ground truth (None if not provided).
    recall_at_k_gt: Option<f64>,
    /// Recall against the full-corpus judge labels (only with --judge-all-pairs).
    recall_at_k_judge: Option<f64>,
    judged: Vec<JudgedBullet>,
}

#[derive(Debug, Serialize)]
struct Report {
    timestamp_unix: u64,
    model: String,
    top_k: i32,
    judge_all_pairs: bool,
    corpus_size: usize,
    num_queries: usize,
    aggregate: Aggregate,
    queries: Vec<QueryResult>,
}

#[derive(Debug, Serialize)]
struct Aggregate {
    mean_precision_at_k: f64,
    mean_ndcg_at_k: f64,
    mean_mrr: f64,
    mean_grade: f64,
    mean_recall_at_k_gt: Option<f64>,
    mean_recall_at_k_judge: Option<f64>,
}

// ─── Metric helpers ───

/// A bullet counts as relevant when its judged grade is >= 2.
const RELEVANT_THRESHOLD: u8 = 2;

fn dcg(grades: &[u8]) -> f64 {
    grades
        .iter()
        .enumerate()
        .map(|(i, &g)| ((2f64.powi(g as i32)) - 1.0) / ((i as f64) + 2.0).log2())
        .sum()
}

/// nDCG of the retrieved ordering vs. the ideal ordering of the grades we have.
/// In default mode `ideal_pool` is the retrieved grades; with --judge-all-pairs it
/// is the full-corpus grades (a stricter, more standard nDCG@k).
fn ndcg(retrieved_grades: &[u8], ideal_pool: &[u8], k: usize) -> f64 {
    let actual = dcg(retrieved_grades);
    let mut ideal_sorted = ideal_pool.to_vec();
    ideal_sorted.sort_unstable_by(|a, b| b.cmp(a));
    ideal_sorted.truncate(k);
    let ideal = dcg(&ideal_sorted);
    if ideal > 0.0 {
        actual / ideal
    } else {
        0.0
    }
}

fn vec_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Build the LLM-judge prompt for a single (job description, bullet) pair.
fn judge_prompt(job_description: &str, bullet: &str) -> String {
    format!(
        "You are evaluating whether a resume bullet point is relevant to a job \
description. Grade the relevance on a 0-3 scale:\n\
  0 = irrelevant\n\
  1 = slightly related\n\
  2 = relevant\n\
  3 = highly relevant / strong match\n\n\
Respond with ONLY a single digit (0, 1, 2, or 3) and nothing else.\n\n\
Job description:\n{}\n\nResume bullet:\n{}\n\nRelevance (0-3):",
        job_description.trim(),
        bullet.trim()
    )
}

/// Extract the first 0-3 digit from the model's reply.
fn parse_grade(reply: &str) -> Option<u8> {
    reply
        .chars()
        .find(|c| ('0'..='3').contains(c))
        .and_then(|c| c.to_digit(10))
        .map(|d| d as u8)
}

fn main() {
    let cfg = parse_args();

    // ── Load dataset ──
    let raw = std::fs::read_to_string(&cfg.dataset)
        .unwrap_or_else(|e| panic!("Failed to read dataset {:?}: {}", cfg.dataset, e));
    let dataset: Dataset =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("Invalid dataset JSON: {}", e));

    println!(
        "Loaded {} corpus bullets, {} queries from {:?}",
        dataset.corpus.len(),
        dataset.queries.len(),
        cfg.dataset
    );

    // ── Load embedding model (same path as the app's resources/model) ──
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("resources")
        .join("model");
    println!("Loading embedding model from {:?} ...", model_dir);
    let mut model = EmbeddingModel::load(&model_dir).expect("Failed to load embedding model");

    // ── Seed a throwaway DB with the corpus and embed every bullet ──
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let db_path = std::env::temp_dir().join(format!("rag_eval_{}.db", ts));
    let db_path_str = db_path.to_str().expect("invalid temp db path").to_string();
    let conn = init_db(&db_path_str).expect("Failed to init eval DB");

    let mut id_to_key: HashMap<i64, String> = HashMap::new();
    for b in &dataset.corpus {
        conn.execute(
            "INSERT INTO experiences (title, category) VALUES (?1, 'job')",
            rusqlite::params![if b.experience.is_empty() { &b.key } else { &b.experience }],
        )
        .expect("insert experience");
        let exp_id = conn.last_insert_rowid();

        conn.execute(
            "INSERT INTO bullet_points (experience_id, content, sort_order) VALUES (?1, ?2, 0)",
            rusqlite::params![exp_id, b.content],
        )
        .expect("insert bullet");
        let bullet_id = conn.last_insert_rowid();

        let embedding = model.embed(&b.content).expect("embed bullet");
        conn.execute(
            "INSERT OR REPLACE INTO bullet_embeddings (bullet_id, embedding) VALUES (?1, ?2)",
            rusqlite::params![bullet_id, vec_to_bytes(&embedding)],
        )
        .expect("store embedding");

        id_to_key.insert(bullet_id, b.key.clone());
    }
    println!("Seeded and embedded {} bullets.\n", id_to_key.len());

    // ── Run retrieval + judging per query ──
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let mut query_results: Vec<QueryResult> = Vec::new();

    for q in &dataset.queries {
        let retrieved: Vec<ScoredBullet> =
            search_similar_conn(&conn, &mut model, &q.job_description, None, cfg.top_k)
                .expect("retrieval failed");

        // Judge each retrieved bullet (graded 0-3).
        let mut judged: Vec<JudgedBullet> = Vec::new();
        for (rank, sb) in retrieved.iter().enumerate() {
            let grade = judge(&rt, &cfg, &q.job_description, &sb.content);
            judged.push(JudgedBullet {
                key: id_to_key.get(&sb.id).cloned().unwrap_or_default(),
                rank: rank + 1,
                distance: sb.distance,
                grade,
                content: sb.content.clone(),
            });
        }

        let retrieved_grades: Vec<u8> = judged.iter().map(|j| j.grade).collect();
        let k = cfg.top_k as usize;
        let n = retrieved_grades.len();

        // Optionally judge the rest of the corpus for true full-corpus recall/nDCG.
        let mut full_grades: Vec<u8> = retrieved_grades.clone();
        let mut recall_at_k_judge: Option<f64> = None;
        if cfg.judge_all_pairs {
            let retrieved_ids: std::collections::HashSet<i64> =
                retrieved.iter().map(|s| s.id).collect();
            let mut relevant_total = retrieved_grades
                .iter()
                .filter(|&&g| g >= RELEVANT_THRESHOLD)
                .count();
            for b in &dataset.corpus {
                // Skip corpus bullets that were already retrieved (and judged).
                let already = id_to_key
                    .iter()
                    .any(|(id, key)| *key == b.key && retrieved_ids.contains(id));
                if already {
                    continue;
                }
                let g = judge(&rt, &cfg, &q.job_description, &b.content);
                full_grades.push(g);
                if g >= RELEVANT_THRESHOLD {
                    relevant_total += 1;
                }
            }
            let relevant_retrieved = retrieved_grades
                .iter()
                .filter(|&&g| g >= RELEVANT_THRESHOLD)
                .count();
            recall_at_k_judge = Some(if relevant_total > 0 {
                relevant_retrieved as f64 / relevant_total as f64
            } else {
                f64::NAN
            });
        }

        let relevant_retrieved = retrieved_grades
            .iter()
            .filter(|&&g| g >= RELEVANT_THRESHOLD)
            .count();
        let precision_at_k = if n > 0 {
            relevant_retrieved as f64 / n as f64
        } else {
            0.0
        };
        let ndcg_at_k = ndcg(&retrieved_grades, &full_grades, k);
        let mrr = retrieved_grades
            .iter()
            .position(|&g| g >= RELEVANT_THRESHOLD)
            .map(|p| 1.0 / (p as f64 + 1.0))
            .unwrap_or(0.0);
        let mean_grade = if n > 0 {
            retrieved_grades.iter().map(|&g| g as f64).sum::<f64>() / n as f64
        } else {
            0.0
        };

        // Ground-truth recall (deterministic, no LLM) when relevant_keys given.
        let recall_at_k_gt = if q.relevant_keys.is_empty() {
            None
        } else {
            let retrieved_keys: std::collections::HashSet<&str> =
                judged.iter().map(|j| j.key.as_str()).collect();
            let hits = q
                .relevant_keys
                .iter()
                .filter(|k| retrieved_keys.contains(k.as_str()))
                .count();
            Some(hits as f64 / q.relevant_keys.len() as f64)
        };

        // Per-query console summary.
        println!("─ {} ─", q.id);
        for j in &judged {
            println!(
                "   [{}] grade={} dist={:.3} {:<14} {}",
                j.rank,
                j.grade,
                j.distance,
                truncate(&j.key, 14),
                truncate(&j.content, 70)
            );
        }
        println!(
            "   P@{k}={:.2}  nDCG@{k}={:.2}  MRR={:.2}  meanGrade={:.2}{}{}\n",
            precision_at_k,
            ndcg_at_k,
            mrr,
            mean_grade,
            recall_at_k_gt
                .map(|r| format!("  Recall@{k}(gt)={:.2}", r))
                .unwrap_or_default(),
            recall_at_k_judge
                .map(|r| format!("  Recall@{k}(judge)={:.2}", r))
                .unwrap_or_default(),
            k = k
        );

        query_results.push(QueryResult {
            id: q.id.clone(),
            retrieved: n,
            precision_at_k,
            ndcg_at_k,
            mrr,
            mean_grade,
            recall_at_k_gt,
            recall_at_k_judge,
            judged,
        });
    }

    // ── Aggregate ──
    let nq = query_results.len().max(1) as f64;
    let mean = |f: &dyn Fn(&QueryResult) -> f64| -> f64 {
        query_results.iter().map(|q| f(q)).sum::<f64>() / nq
    };
    let mean_opt = |f: &dyn Fn(&QueryResult) -> Option<f64>| -> Option<f64> {
        let vals: Vec<f64> = query_results
            .iter()
            .filter_map(|q| f(q))
            .filter(|v| !v.is_nan())
            .collect();
        if vals.is_empty() {
            None
        } else {
            Some(vals.iter().sum::<f64>() / vals.len() as f64)
        }
    };

    let aggregate = Aggregate {
        mean_precision_at_k: mean(&|q| q.precision_at_k),
        mean_ndcg_at_k: mean(&|q| q.ndcg_at_k),
        mean_mrr: mean(&|q| q.mrr),
        mean_grade: mean(&|q| q.mean_grade),
        mean_recall_at_k_gt: mean_opt(&|q| q.recall_at_k_gt),
        mean_recall_at_k_judge: mean_opt(&|q| q.recall_at_k_judge),
    };

    println!("══ Aggregate (over {} queries) ══", query_results.len());
    println!("   mean P@{}        = {:.3}", cfg.top_k, aggregate.mean_precision_at_k);
    println!("   mean nDCG@{}     = {:.3}", cfg.top_k, aggregate.mean_ndcg_at_k);
    println!("   mean MRR         = {:.3}", aggregate.mean_mrr);
    println!("   mean grade       = {:.3}", aggregate.mean_grade);
    if let Some(r) = aggregate.mean_recall_at_k_gt {
        println!("   mean Recall@{}(gt)    = {:.3}", cfg.top_k, r);
    }
    if let Some(r) = aggregate.mean_recall_at_k_judge {
        println!("   mean Recall@{}(judge) = {:.3}", cfg.top_k, r);
    }

    // ── Write report ──
    let report = Report {
        timestamp_unix: ts,
        model: cfg.model.clone(),
        top_k: cfg.top_k,
        judge_all_pairs: cfg.judge_all_pairs,
        corpus_size: dataset.corpus.len(),
        num_queries: query_results.len(),
        aggregate,
        queries: query_results,
    };

    std::fs::create_dir_all(&cfg.out_dir).expect("create results dir");
    let out_path = cfg.out_dir.join(format!("{}.json", ts));
    std::fs::write(
        &out_path,
        serde_json::to_string_pretty(&report).expect("serialize report"),
    )
    .expect("write report");
    println!("\nWrote report to {:?}", out_path);

    // Best-effort cleanup of the throwaway DB (and WAL/SHM sidecars).
    let _ = std::fs::remove_file(&db_path);
    let _ = std::fs::remove_file(format!("{}-wal", db_path_str));
    let _ = std::fs::remove_file(format!("{}-shm", db_path_str));
}

/// Run one judge call synchronously on the tokio runtime, returning a 0-3 grade.
fn judge(rt: &tokio::runtime::Runtime, cfg: &Config, jd: &str, bullet: &str) -> u8 {
    let prompt = judge_prompt(jd, bullet);
    // temperature 0 + tiny num_predict for a deterministic single-digit answer.
    let reply = rt.block_on(generate_ollama_opts(&prompt, &cfg.model, &cfg.ollama_url, 0.0, 8));
    match reply {
        Ok(text) => parse_grade(&text).unwrap_or_else(|| {
            eprintln!("   ! could not parse grade from reply {:?}, defaulting to 0", text);
            0
        }),
        Err(e) => {
            eprintln!("   ! judge call failed: {} — defaulting to grade 0", e);
            0
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}
