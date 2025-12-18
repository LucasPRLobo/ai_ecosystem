/**
 * Prompted Town Web UI - Client JavaScript
 */

// =============================================================================
// Socket Connection
// =============================================================================

const socket = io();

// Connection status
socket.on('connect', () => {
    console.log('Connected to server');
    updateConnectionStatus(true);
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    updateConnectionStatus(false);
});

function updateConnectionStatus(connected) {
    const indicator = document.getElementById('connection-status');
    if (connected) {
        indicator.className = 'connection-indicator connected';
        indicator.innerHTML = '<span class="dot"></span> Connected';
    } else {
        indicator.className = 'connection-indicator disconnected';
        indicator.innerHTML = '<span class="dot"></span> Disconnected';
    }
}

// =============================================================================
// State Updates
// =============================================================================

socket.on('state_update', (state) => {
    updateUI(state);
});

socket.on('step_complete', (state) => {
    updateUI(state);

    // Handle conversations from this step
    if (state.step_conversations && state.step_conversations.length > 0) {
        state.step_conversations.forEach(conv => {
            addConversationEntry(conv);
            // Highlight conversation on graph
            if (window.townGraph) {
                window.townGraph.highlightConversation(conv.initiator, conv.target);
            }
        });
    }

    // Handle arrests from this step
    if (state.step_arrests && state.step_arrests.length > 0) {
        state.step_arrests.forEach(arrest => {
            addArrestEntry(arrest);
        });
    }
});

socket.on('started', () => {
    setRunningState(true);
});

socket.on('stopped', () => {
    setRunningState(false);
});

socket.on('episode_ended', () => {
    setRunningState(false);
    addLogEntry({
        type: 'system',
        message: 'Episode ended'
    });
});

socket.on('conversation', (data) => {
    if (data.conversations) {
        data.conversations.forEach(conv => {
            addConversationEntry(conv);
            // Highlight conversation on graph
            if (window.townGraph) {
                window.townGraph.highlightConversation(conv.initiator, conv.target);
            }
        });
    }
    if (data.state) {
        updateUI(data.state);
    }
});

socket.on('agent_thoughts', (data) => {
    if (data.thoughts && data.thoughts.length > 0) {
        data.thoughts.forEach(thought => {
            addThoughtEntry(thought);
        });
    }
});

// =============================================================================
// UI Update Functions
// =============================================================================

let currentAgents = [];

function updateUI(state) {
    if (!state || Object.keys(state).length === 0) return;

    // Update stats
    document.getElementById('stat-day').textContent = state.day || 1;
    document.getElementById('stat-tick').textContent = state.tick || 0;
    document.getElementById('stat-time').textContent = state.time_of_day || 'MORNING';
    document.getElementById('stat-recruited').textContent = state.total_recruited || 0;
    document.getElementById('stat-arrests').textContent = state.total_arrests || 0;
    document.getElementById('stat-convs').textContent = state.total_conversations || 0;

    // Store agents for both views
    currentAgents = state.agents || [];

    // Update agents list view
    updateAgentsList(currentAgents);

    // Update graph view
    if (window.townGraph) {
        window.townGraph.updateAgents(currentAgents);
    }

    // Update AI status
    const aiStatus = document.getElementById('ai-status');
    const aiToggle = document.getElementById('ai-toggle');
    if (state.use_ai) {
        aiStatus.textContent = 'ON';
        aiStatus.className = 'status-badge status-on';
        aiToggle.checked = true;
    } else {
        aiStatus.textContent = 'OFF';
        aiStatus.className = 'status-badge status-off';
        aiToggle.checked = false;
    }
}

function updateAgentsList(agents) {
    const container = document.getElementById('agents-container');
    container.innerHTML = '';

    agents.forEach(agent => {
        const card = createAgentCard(agent);
        container.appendChild(card);
    });
}

function createAgentCard(agent) {
    const card = document.createElement('div');
    card.className = 'agent-card';

    if (agent.is_arrested) card.classList.add('arrested');
    if (agent.is_recruited) card.classList.add('recruited');

    // Get role icon
    const roleIcons = {
        'farmer': '🌾',
        'guard': '🛡️',
        'rebel': '⚔️'
    };
    const icon = roleIcons[agent.role] || '👤';

    card.innerHTML = `
        <div class="agent-header">
            <div class="agent-avatar ${agent.role}">${icon}</div>
            <div class="agent-info">
                <h3>${agent.name}</h3>
                <div class="agent-role">${agent.role}</div>
            </div>
        </div>
        <div class="agent-details">
            <div class="agent-detail">
                <span>Energy</span>
                <span>${agent.energy}</span>
            </div>
            <div class="agent-detail">
                <span>Gold</span>
                <span>${agent.gold}</span>
            </div>
            <div class="agent-detail">
                <span>Suspicion</span>
                <span>${(agent.suspicion * 100).toFixed(0)}%</span>
            </div>
            <div class="agent-detail">
                <span>Can Act</span>
                <span>${agent.can_act ? '✓' : '✗'}</span>
            </div>
        </div>
        <div class="agent-location">
            📍 ${formatLocation(agent.location)}
        </div>
        ${getStatusTags(agent)}
    `;

    // Add click handler to open editor
    card.style.cursor = 'pointer';
    card.addEventListener('click', () => {
        openAgentEditor(agent.id);
    });

    return card;
}

function formatLocation(location) {
    if (!location) return 'Unknown';
    return location.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function getStatusTags(agent) {
    const tags = [];

    if (agent.true_faction === 'rebel' && agent.role !== 'rebel') {
        tags.push('<span class="status-tag rebel">Secret Rebel</span>');
    }
    if (agent.is_recruited) {
        tags.push('<span class="status-tag recruited">Recruited</span>');
    }
    if (agent.is_arrested) {
        tags.push('<span class="status-tag arrested">Arrested</span>');
    }

    if (tags.length === 0) return '';
    return `<div class="agent-status">${tags.join('')}</div>`;
}

// =============================================================================
// View Toggle
// =============================================================================

let currentView = 'graph';

function setView(view) {
    currentView = view;
    const graphContainer = document.getElementById('graph-container');
    const agentsContainer = document.getElementById('agents-container');
    const btnGraph = document.getElementById('btn-graph-view');
    const btnList = document.getElementById('btn-list-view');

    if (view === 'graph') {
        graphContainer.style.display = 'block';
        agentsContainer.style.display = 'none';
        btnGraph.classList.add('active');
        btnList.classList.remove('active');

        // Trigger resize to fix graph dimensions
        if (window.townGraph) {
            window.townGraph.updateDimensions();
            window.townGraph.updateAgents(currentAgents);
        }
    } else {
        graphContainer.style.display = 'none';
        agentsContainer.style.display = 'grid';
        btnGraph.classList.remove('active');
        btnList.classList.add('active');
    }
}

// =============================================================================
// Conversation Log
// =============================================================================

function addConversationEntry(conv) {
    const log = document.getElementById('conversation-log');

    // Remove placeholder if present
    const placeholder = log.querySelector('.log-placeholder');
    if (placeholder) placeholder.remove();

    const entry = document.createElement('div');
    entry.className = 'conversation-entry';

    // Determine entry type
    if (conv.outcome_summary && conv.outcome_summary.includes('recruit')) {
        entry.classList.add('recruitment');
    }

    let messagesHtml = '';
    if (conv.transcript && conv.transcript.length > 0) {
        messagesHtml = '<div class="conversation-messages">';
        conv.transcript.forEach(msg => {
            messagesHtml += `
                <div class="message">
                    <span class="message-speaker">${msg.speaker}:</span>
                    <span class="message-text">${msg.text}</span>
                </div>
            `;
        });
        messagesHtml += '</div>';
    }

    entry.innerHTML = `
        <div class="conversation-header">
            <span class="conversation-participants">${conv.initiator} → ${conv.target}</span>
            <span class="conversation-intent">${conv.intent}</span>
        </div>
        ${messagesHtml}
        <div class="conversation-outcome">${conv.outcome_summary || ''}</div>
    `;

    log.insertBefore(entry, log.firstChild);

    // Limit log entries
    while (log.children.length > 50) {
        log.removeChild(log.lastChild);
    }
}

function addLogEntry(entry) {
    const log = document.getElementById('conversation-log');

    const placeholder = log.querySelector('.log-placeholder');
    if (placeholder) placeholder.remove();

    const div = document.createElement('div');
    div.className = 'conversation-entry';
    div.innerHTML = `<div class="conversation-outcome">${entry.message}</div>`;

    log.insertBefore(div, log.firstChild);
}

function addThoughtEntry(thought) {
    const log = document.getElementById('conversation-log');

    // Remove placeholder if present
    const placeholder = log.querySelector('.log-placeholder');
    if (placeholder) placeholder.remove();

    const entry = document.createElement('div');
    entry.className = 'conversation-entry thought-entry';

    // Get role-based styling class
    const roleClass = thought.role || 'farmer';

    entry.innerHTML = `
        <div class="thought-header">
            <span class="thought-agent ${roleClass}">${thought.agent_name}</span>
            <span class="thought-label">💭 thinking</span>
        </div>
        <div class="thought-content">
            <em>"${thought.thought}"</em>
        </div>
    `;

    log.insertBefore(entry, log.firstChild);

    // Limit log entries
    while (log.children.length > 50) {
        log.removeChild(log.lastChild);
    }
}

function addArrestEntry(arrest) {
    const log = document.getElementById('conversation-log');

    // Remove placeholder if present
    const placeholder = log.querySelector('.log-placeholder');
    if (placeholder) placeholder.remove();

    const entry = document.createElement('div');
    entry.className = 'conversation-entry arrest-entry';

    const suspicionPercent = arrest.suspicion_level
        ? Math.round(arrest.suspicion_level * 100)
        : '??';

    entry.innerHTML = `
        <div class="arrest-header">
            <span class="arrest-icon">⚔️</span>
            <span class="arrest-title">ARREST</span>
        </div>
        <div class="arrest-content">
            <strong>${arrest.guard_name || 'A guard'}</strong> arrested
            <strong>${arrest.arrested_name || arrest.arrested_id}</strong>
            at ${arrest.location || 'unknown location'}
        </div>
        <div class="arrest-details">
            Suspicion level: ${suspicionPercent}%
            ${arrest.reason ? ` • Reason: ${arrest.reason}` : ''}
        </div>
    `;

    log.insertBefore(entry, log.firstChild);

    // Limit log entries
    while (log.children.length > 50) {
        log.removeChild(log.lastChild);
    }
}

// =============================================================================
// Button State
// =============================================================================

function setRunningState(running) {
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');
    const btnStep = document.getElementById('btn-step');

    btnStart.disabled = running;
    btnStop.disabled = !running;
    btnStep.disabled = running;
}

// =============================================================================
// Event Handlers
// =============================================================================

document.getElementById('btn-start').addEventListener('click', () => {
    socket.emit('start');
});

document.getElementById('btn-stop').addEventListener('click', () => {
    socket.emit('stop');
});

document.getElementById('btn-step').addEventListener('click', () => {
    socket.emit('step');
});

document.getElementById('btn-reset').addEventListener('click', () => {
    socket.emit('reset');
    document.getElementById('conversation-log').innerHTML =
        '<div class="log-placeholder">Conversations will appear here...</div>';
});

document.getElementById('btn-force-conv').addEventListener('click', () => {
    socket.emit('force_conversation');
});

document.getElementById('btn-thoughts').addEventListener('click', () => {
    socket.emit('generate_thoughts');
});

document.getElementById('speed-slider').addEventListener('input', (e) => {
    const speed = parseFloat(e.target.value);
    document.getElementById('speed-value').textContent = speed.toFixed(1) + 's';

    fetch('/api/set_speed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed: speed })
    });
});

document.getElementById('ai-toggle').addEventListener('change', (e) => {
    const enable = e.target.checked;

    fetch('/api/toggle_ai', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enable: enable })
    })
    .then(res => res.json())
    .then(data => {
        const aiStatus = document.getElementById('ai-status');
        if (data.ai_enabled) {
            aiStatus.textContent = 'ON';
            aiStatus.className = 'status-badge status-on';
        } else {
            aiStatus.textContent = 'OFF';
            aiStatus.className = 'status-badge status-off';
            e.target.checked = false;
        }

        if (data.message) {
            addLogEntry({ type: 'system', message: data.message });
        }
    });
});

// View toggle buttons
document.getElementById('btn-graph-view').addEventListener('click', () => setView('graph'));
document.getElementById('btn-list-view').addEventListener('click', () => setView('list'));

// =============================================================================
// Agent Editor
// =============================================================================

let currentEditingAgent = null;
let availableTemplates = {};

// Fetch available templates on load
async function loadTemplates() {
    try {
        const res = await fetch('/api/templates');
        const data = await res.json();
        availableTemplates = data.agent_templates || {};

        // Populate template dropdown
        const select = document.getElementById('template-select');
        select.innerHTML = '<option value="">-- Select Template --</option>';
        Object.entries(availableTemplates).forEach(([name, template]) => {
            const displayName = name.replace(/_/g, ' ').replace(/\bthe\b/i, 'The');
            const option = document.createElement('option');
            option.value = name;
            option.textContent = displayName;
            option.title = template.description;
            select.appendChild(option);
        });
    } catch (err) {
        console.error('Failed to load templates:', err);
    }
}

// Open the editor for an agent
async function openAgentEditor(agentId) {
    try {
        const res = await fetch(`/api/agent/${agentId}`);
        if (!res.ok) {
            console.error('Failed to fetch agent details');
            return;
        }

        const agent = await res.json();
        currentEditingAgent = agent;

        // Populate editor fields
        document.getElementById('editor-agent-name').textContent = agent.name;

        const roleBadge = document.getElementById('editor-agent-role');
        roleBadge.textContent = agent.role;
        roleBadge.className = `role-badge ${agent.role}`;

        const factionBadge = document.getElementById('editor-agent-faction');
        factionBadge.textContent = agent.faction;
        factionBadge.className = `faction-badge ${agent.faction}`;

        document.getElementById('editor-agent-description').textContent =
            agent.personality?.core_identity || '';

        // Show/hide recruitment section for rebels
        const recruitSection = document.getElementById('recruitment-section');
        if (agent.faction === 'rebel') {
            recruitSection.style.display = 'block';
            document.getElementById('recruitment-style-select').value =
                agent.recruitment_style || 'moderate';
        } else {
            recruitSection.style.display = 'none';
        }

        // Populate trait sliders
        if (agent.traits) {
            Object.entries(agent.traits).forEach(([trait, value]) => {
                const slider = document.getElementById(`trait-${trait}`);
                const valSpan = document.getElementById(`val-${trait}`);
                if (slider) {
                    slider.value = value;
                    if (valSpan) valSpan.textContent = value.toFixed(1);
                }
            });
        }

        // Update description
        document.getElementById('traits-description').textContent =
            agent.traits_description || 'Balanced personality';

        // Show the modal
        document.getElementById('agent-editor').style.display = 'flex';

    } catch (err) {
        console.error('Error opening agent editor:', err);
    }
}

// Close the editor
function closeAgentEditor() {
    document.getElementById('agent-editor').style.display = 'none';
    currentEditingAgent = null;
}

// Save trait changes
async function saveTraits() {
    if (!currentEditingAgent) return;

    const traits = {};
    const traitNames = ['boldness', 'discretion', 'warmth', 'trust', 'compliance', 'ambition', 'resilience', 'grievance'];

    traitNames.forEach(trait => {
        const slider = document.getElementById(`trait-${trait}`);
        if (slider) {
            traits[trait] = parseFloat(slider.value);
        }
    });

    try {
        const res = await fetch(`/api/agent/${currentEditingAgent.id}/traits`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ traits })
        });

        const data = await res.json();
        if (data.status === 'ok') {
            document.getElementById('traits-description').textContent = data.description;
            addLogEntry({ type: 'system', message: `Updated traits for ${currentEditingAgent.name}` });
        }
    } catch (err) {
        console.error('Error saving traits:', err);
    }
}

// Apply a template
async function applyTemplate() {
    if (!currentEditingAgent) return;

    const templateName = document.getElementById('template-select').value;
    if (!templateName) return;

    try {
        const res = await fetch(`/api/agent/${currentEditingAgent.id}/template`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ template: templateName })
        });

        const data = await res.json();
        if (data.status === 'ok') {
            // Update sliders with new values
            Object.entries(data.traits).forEach(([trait, value]) => {
                const slider = document.getElementById(`trait-${trait}`);
                const valSpan = document.getElementById(`val-${trait}`);
                if (slider) {
                    slider.value = value;
                    if (valSpan) valSpan.textContent = value.toFixed(1);
                }
            });
            document.getElementById('traits-description').textContent = data.description;
            addLogEntry({ type: 'system', message: `Applied "${templateName.replace(/_/g, ' ')}" template to ${currentEditingAgent.name}` });
        }
    } catch (err) {
        console.error('Error applying template:', err);
    }
}

// Update recruitment style
async function updateRecruitmentStyle() {
    if (!currentEditingAgent) return;

    const style = document.getElementById('recruitment-style-select').value;

    try {
        const res = await fetch(`/api/agent/${currentEditingAgent.id}/recruitment_style`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ style })
        });

        const data = await res.json();
        if (data.status === 'ok') {
            addLogEntry({ type: 'system', message: `Set ${currentEditingAgent.name}'s recruitment style to ${style}` });
        }
    } catch (err) {
        console.error('Error updating recruitment style:', err);
    }
}

// Event listeners for editor
document.getElementById('editor-close').addEventListener('click', closeAgentEditor);
document.getElementById('btn-save-traits').addEventListener('click', saveTraits);
document.getElementById('btn-apply-template').addEventListener('click', applyTemplate);
document.getElementById('recruitment-style-select').addEventListener('change', updateRecruitmentStyle);

// Close modal when clicking overlay
document.getElementById('agent-editor').addEventListener('click', (e) => {
    if (e.target.id === 'agent-editor') {
        closeAgentEditor();
    }
});

// Update slider value displays
document.querySelectorAll('.trait-slider').forEach(slider => {
    slider.addEventListener('input', (e) => {
        const trait = e.target.id.replace('trait-', '');
        const valSpan = document.getElementById(`val-${trait}`);
        if (valSpan) {
            valSpan.textContent = parseFloat(e.target.value).toFixed(1);
        }
    });
});

// Listen for agent selection from graph
window.addEventListener('agentSelected', (e) => {
    openAgentEditor(e.detail.agentId);
});

// =============================================================================
// Initialize
// =============================================================================

// Initialize graph after DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Load templates for agent editor
    loadTemplates();

    // Small delay to ensure graph container is sized
    setTimeout(() => {
        if (!window.townGraph) {
            window.townGraph = new TownGraph('graph-container', 'town-graph');
        }

        // Fetch initial state
        fetch('/api/state')
            .then(res => res.json())
            .then(state => {
                updateUI(state);
            });

        // Fetch conversation history
        fetch('/api/conversations')
            .then(res => res.json())
            .then(conversations => {
                conversations.forEach(conv => addConversationEntry(conv));
            });
    }, 100);
});
