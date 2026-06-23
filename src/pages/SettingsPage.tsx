import { useState, useEffect } from 'react';
import { getLlmSettings, updateLlmSettings, LLMSettings } from '../lib/tauri';

export default function SettingsPage() {
    const [settings, setSettings] = useState<LLMSettings>({ mode: 'local', ollama_model: 'qwen3.5:9b', ollama_url: 'http://localhost:11434', api_key: '' });

    useEffect(() => {
        getLlmSettings().then(setSettings).catch(console.error);
    }, []);

    const handleSave = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            await updateLlmSettings(settings.mode, settings.ollama_model, settings.api_key, settings.ollama_url, settings.cloud_model);
            alert('Settings saved!');
        } catch (error) {
            console.error('Failed to save settings', error);
            alert('Failed to save settings: ' + error);
        }
    };

    return (
        <div>
            <h1>Settings</h1>
            <form onSubmit={handleSave} style={{ display: 'flex', flexDirection: 'column', gap: '1rem', maxWidth: '400px' }}>
                <div>
                    <label style={{ display: 'block', marginBottom: '0.5rem' }}>Mode:</label>
                    <select value={settings.mode} onChange={e => setSettings({ ...settings, mode: e.target.value })} style={{ width: '100%', padding: '0.5rem' }}>
                        <option value="local">Local (Ollama)</option>
                        <option value="cloud">Cloud API</option>
                    </select>
                </div>

                {settings.mode === 'local' && (
                    <>
                        <div>
                            <label style={{ display: 'block', marginBottom: '0.5rem' }}>Ollama Model:</label>
                            <input
                                placeholder="qwen3.5:9b"
                                value={settings.ollama_model || ''}
                                onChange={e => setSettings({ ...settings, ollama_model: e.target.value })}
                                style={{ width: '100%', padding: '0.5rem' }}
                            />
                            <small style={{ color: '#888' }}>Must be pulled locally: <code>ollama pull {settings.ollama_model || 'qwen3.5:9b'}</code></small>
                        </div>
                        <div>
                            <label style={{ display: 'block', marginBottom: '0.5rem' }}>Ollama Server URL:</label>
                            <input
                                placeholder="http://localhost:11434"
                                value={settings.ollama_url || ''}
                                onChange={e => setSettings({ ...settings, ollama_url: e.target.value })}
                                style={{ width: '100%', padding: '0.5rem' }}
                            />
                        </div>
                    </>
                )}

                {settings.mode === 'cloud' && (
                    <div>
                        <label style={{ display: 'block', marginBottom: '0.5rem' }}>API Key:</label>
                        <input
                            type="password"
                            value={settings.api_key || ''}
                            onChange={e => setSettings({ ...settings, api_key: e.target.value })}
                            style={{ width: '100%', padding: '0.5rem' }}
                        />
                    </div>
                )}

                <button type="submit" style={{ padding: '0.5rem' }}>Save Settings</button>
            </form>
        </div>
    );
}
