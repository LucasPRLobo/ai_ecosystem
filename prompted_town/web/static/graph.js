/**
 * Prompted Town - D3.js Graph Visualization
 *
 * Shows locations as large nodes and agents connected to their current location.
 * Agents are smaller nodes orbiting around their location.
 */

class TownGraph {
    constructor(containerId, svgId) {
        this.container = document.getElementById(containerId);
        this.svg = d3.select(`#${svgId}`);
        this.width = 0;
        this.height = 0;
        this.simulation = null;
        this.nodes = [];
        this.links = [];

        // Location definitions with fixed positions (will be scaled)
        // Actual locations from core/types.py: farm, market, tavern, home, gate, town_square
        // Note: 'home' is not in this list - individual homes are created dynamically per agent
        // 'jail' is a visual-only location for arrested agents
        this.locations = [
            { id: 'town_square', name: 'Town Square', x: 0.5, y: 0.5 },
            { id: 'market', name: 'Market', x: 0.25, y: 0.4 },
            { id: 'tavern', name: 'Tavern', x: 0.75, y: 0.4 },
            { id: 'farm', name: 'Farm', x: 0.2, y: 0.75 },
            { id: 'gate', name: 'Gate', x: 0.8, y: 0.75 },
            { id: 'jail', name: 'Jail', x: 0.5, y: 0.88, isJail: true },
        ];

        // Individual agent homes - positioned in residential area at top
        // These are generated dynamically based on agents
        this.agentHomes = {};

        // Location connections (roads)
        this.roads = [
            ['town_square', 'market'],
            ['town_square', 'tavern'],
            ['market', 'farm'],
            ['tavern', 'gate'],
            ['farm', 'town_square'],
            ['gate', 'town_square'],
        ];

        this.tooltip = document.getElementById('tooltip');
        this.init();
    }

    init() {
        this.updateDimensions();
        window.addEventListener('resize', () => this.updateDimensions());
        this.createGraph();
    }

    updateDimensions() {
        const rect = this.container.getBoundingClientRect();
        this.width = rect.width;
        this.height = rect.height;

        this.svg
            .attr('width', this.width)
            .attr('height', this.height);

        if (this.simulation) {
            this.updateNodePositions();
        }
    }

    createGraph() {
        // Create arrow marker for directed edges
        this.svg.append('defs').append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '-0 -5 10 10')
            .attr('refX', 20)
            .attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .append('path')
            .attr('d', 'M 0,-5 L 10,0 L 0,5')
            .attr('fill', '#4a4a6a');

        // Create groups for different layers
        this.roadGroup = this.svg.append('g').attr('class', 'roads');
        this.linkGroup = this.svg.append('g').attr('class', 'links');
        this.nodeGroup = this.svg.append('g').attr('class', 'nodes');
        this.labelGroup = this.svg.append('g').attr('class', 'labels');

        // Draw roads between locations
        this.drawRoads();

        // Create location nodes
        this.drawLocations();
    }

    drawRoads() {
        const roadData = this.roads.map(([from, to]) => {
            const fromLoc = this.locations.find(l => l.id === from);
            const toLoc = this.locations.find(l => l.id === to);
            return { from: fromLoc, to: toLoc };
        });

        this.roadGroup.selectAll('line')
            .data(roadData)
            .enter()
            .append('line')
            .attr('class', 'road')
            .attr('x1', d => d.from.x * this.width)
            .attr('y1', d => d.from.y * this.height)
            .attr('x2', d => d.to.x * this.width)
            .attr('y2', d => d.to.y * this.height)
            .attr('stroke', '#2a2a4a')
            .attr('stroke-width', 3)
            .attr('stroke-dasharray', '10,5');
    }

    drawLocations() {
        const locationNodes = this.nodeGroup.selectAll('.location-node')
            .data(this.locations)
            .enter()
            .append('g')
            .attr('class', d => `location-group ${d.isJail ? 'jail-location' : ''}`)
            .attr('data-location', d => d.id)
            .attr('transform', d => `translate(${d.x * this.width}, ${d.y * this.height})`);

        // Location circles
        locationNodes.append('circle')
            .attr('class', 'location-node')
            .attr('r', 35)
            .attr('fill', d => d.isJail ? '#2d1a1a' : '#1e1e3a')
            .attr('stroke', d => d.isJail ? '#991b1b' : '#4a4a7a')
            .attr('stroke-width', d => d.isJail ? 3 : 2);

        // Location icons
        const icons = {
            'town_square': '🏛️',
            'market': '🏪',
            'tavern': '🍺',
            'farm': '🌾',
            'home': '🏠',
            'gate': '🚪',
            'jail': '⛓️',
        };

        locationNodes.append('text')
            .attr('class', 'location-icon')
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'central')
            .attr('font-size', '20px')
            .text(d => icons[d.id] || '📍');

        // Location labels
        this.labelGroup.selectAll('.location-label')
            .data(this.locations)
            .enter()
            .append('text')
            .attr('class', 'location-label')
            .attr('x', d => d.x * this.width)
            .attr('y', d => d.y * this.height + 50)
            .attr('text-anchor', 'middle')
            .attr('fill', '#8888aa')
            .attr('font-size', '11px')
            .text(d => d.name);
    }

    updateAgents(agents) {
        if (!agents || agents.length === 0) return;

        // Create individual homes for agents if not already created
        this.ensureAgentHomes(agents);

        // Group agents by location (treating 'home' and 'jail' specially)
        const agentsByLocation = {};
        agents.forEach(agent => {
            let loc;
            if (agent.is_arrested) {
                loc = 'jail';
            } else if (agent.location === 'home') {
                loc = `home_${agent.id}`;
            } else {
                loc = agent.location;
            }
            if (!agentsByLocation[loc]) {
                agentsByLocation[loc] = [];
            }
            agentsByLocation[loc].push(agent);
        });

        // Create nodes and links data
        this.nodes = [];
        this.links = [];

        // Add location nodes (fixed positions)
        this.locations.forEach(loc => {
            this.nodes.push({
                id: loc.id,
                type: 'location',
                name: loc.name,
                fx: loc.x * this.width,
                fy: loc.y * this.height,
            });
        });

        // Add individual home nodes
        Object.values(this.agentHomes).forEach(home => {
            this.nodes.push({
                id: home.id,
                type: 'home',
                name: home.name,
                agentId: home.agentId,
                fx: home.x * this.width,
                fy: home.y * this.height,
            });
        });

        // Add agent nodes
        agents.forEach((agent, i) => {
            let locationX, locationY, targetLocationId;

            if (agent.is_arrested) {
                // Arrested agents go to jail
                const jail = this.locations.find(l => l.id === 'jail');
                if (jail) {
                    locationX = jail.x * this.width;
                    locationY = jail.y * this.height;
                    targetLocationId = 'jail';
                } else {
                    return;
                }
            } else if (agent.location === 'home') {
                // Agent is at their personal home
                const home = this.agentHomes[agent.id];
                if (home) {
                    locationX = home.x * this.width;
                    locationY = home.y * this.height;
                    targetLocationId = home.id;
                } else {
                    return; // Skip if no home found
                }
            } else {
                // Agent is at a public location
                const location = this.locations.find(l => l.id === agent.location);
                if (location) {
                    locationX = location.x * this.width;
                    locationY = location.y * this.height;
                    targetLocationId = agent.location;
                } else {
                    return; // Skip unknown locations
                }
            }

            // Calculate position around the location
            let locKey;
            if (agent.is_arrested) {
                locKey = 'jail';
            } else if (agent.location === 'home') {
                locKey = `home_${agent.id}`;
            } else {
                locKey = agent.location;
            }
            const agentsAtLoc = agentsByLocation[locKey] || [agent];
            const index = agentsAtLoc.indexOf(agent);
            const total = agentsAtLoc.length;
            const angle = (2 * Math.PI * index / total) - Math.PI / 2;
            // Smaller radius for home and jail
            let radius;
            if (agent.is_arrested) {
                radius = 35 + (total > 2 ? 10 : 0);
            } else if (agent.location === 'home') {
                radius = 30;
            } else {
                radius = 60 + (total > 4 ? 15 : 0);
            }

            this.nodes.push({
                id: agent.id,
                type: 'agent',
                role: agent.role,
                name: agent.name,
                agent: agent,
                x: locationX + Math.cos(angle) * radius,
                y: locationY + Math.sin(angle) * radius,
                targetX: locationX + Math.cos(angle) * radius,
                targetY: locationY + Math.sin(angle) * radius,
            });

            // Link agent to their current location (home or public)
            this.links.push({
                source: agent.id,
                target: targetLocationId,
            });
        });

        this.updateVisualization();
        this.updateHomeNodes();
    }

    ensureAgentHomes(agents) {
        // Create homes for new agents
        const existingIds = Object.keys(this.agentHomes);
        const newAgents = agents.filter(a => !existingIds.includes(a.id));

        if (newAgents.length === 0 && existingIds.length > 0) return;

        // Position homes in a row at the top of the graph
        const totalAgents = agents.length;
        const homeAreaStartX = 0.15;
        const homeAreaEndX = 0.85;
        const homeY = 0.12;

        agents.forEach((agent, i) => {
            if (!this.agentHomes[agent.id]) {
                const xSpread = homeAreaEndX - homeAreaStartX;
                const xPos = homeAreaStartX + (xSpread * i / Math.max(totalAgents - 1, 1));

                this.agentHomes[agent.id] = {
                    id: `home_${agent.id}`,
                    agentId: agent.id,
                    name: `${agent.name}'s Home`,
                    x: totalAgents === 1 ? 0.5 : xPos,
                    y: homeY,
                    role: agent.role,
                };
            }
        });

        // Draw home nodes if this is the first time
        if (newAgents.length > 0 || existingIds.length === 0) {
            this.drawHomeNodes();
        }
    }

    drawHomeNodes() {
        // Remove existing home nodes
        this.nodeGroup.selectAll('.home-group').remove();
        this.labelGroup.selectAll('.home-label').remove();

        const homes = Object.values(this.agentHomes);

        // Draw home nodes
        const homeNodes = this.nodeGroup.selectAll('.home-group')
            .data(homes, d => d.id)
            .enter()
            .append('g')
            .attr('class', 'home-group')
            .attr('transform', d => `translate(${d.x * this.width}, ${d.y * this.height})`);

        // Home shape (smaller than public locations)
        homeNodes.append('rect')
            .attr('class', 'home-node')
            .attr('x', -18)
            .attr('y', -18)
            .attr('width', 36)
            .attr('height', 36)
            .attr('rx', 6)
            .attr('fill', '#1a1a2e')
            .attr('stroke', d => this.getHomeStroke(d))
            .attr('stroke-width', 2);

        // Home icon
        homeNodes.append('text')
            .attr('class', 'home-icon')
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'central')
            .attr('font-size', '16px')
            .text('🏠');

        // Home labels
        this.labelGroup.selectAll('.home-label')
            .data(homes, d => d.id)
            .enter()
            .append('text')
            .attr('class', 'home-label')
            .attr('x', d => d.x * this.width)
            .attr('y', d => d.y * this.height + 32)
            .attr('text-anchor', 'middle')
            .attr('fill', '#6a6a8a')
            .attr('font-size', '9px')
            .text(d => d.name.replace("'s Home", ""));
    }

    updateHomeNodes() {
        const homes = Object.values(this.agentHomes);

        this.nodeGroup.selectAll('.home-group')
            .data(homes, d => d.id)
            .attr('transform', d => `translate(${d.x * this.width}, ${d.y * this.height})`);

        this.labelGroup.selectAll('.home-label')
            .data(homes, d => d.id)
            .attr('x', d => d.x * this.width)
            .attr('y', d => d.y * this.height + 32);
    }

    getHomeStroke(home) {
        const colors = {
            'farmer': '#4a7024',
            'guard': '#2563eb',
            'rebel': '#b91c1c',
        };
        return colors[home.role] || '#4a4a7a';
    }

    updateVisualization() {
        // Update links
        const linkSelection = this.linkGroup.selectAll('.agent-link')
            .data(this.links, d => `${d.source}-${d.target}`);

        linkSelection.exit().remove();

        linkSelection.enter()
            .append('line')
            .attr('class', 'agent-link')
            .attr('stroke', '#3a3a5a')
            .attr('stroke-width', 1.5)
            .attr('stroke-opacity', 0.6)
            .merge(linkSelection)
            .transition()
            .duration(300)
            .attr('x1', d => {
                const source = this.nodes.find(n => n.id === d.source);
                return source ? (source.x || source.fx) : 0;
            })
            .attr('y1', d => {
                const source = this.nodes.find(n => n.id === d.source);
                return source ? (source.y || source.fy) : 0;
            })
            .attr('x2', d => {
                const target = this.nodes.find(n => n.id === d.target);
                return target ? (target.x || target.fx) : 0;
            })
            .attr('y2', d => {
                const target = this.nodes.find(n => n.id === d.target);
                return target ? (target.y || target.fy) : 0;
            });

        // Update agent nodes
        const agentNodes = this.nodes.filter(n => n.type === 'agent');

        const nodeSelection = this.nodeGroup.selectAll('.agent-node')
            .data(agentNodes, d => d.id);

        nodeSelection.exit()
            .transition()
            .duration(200)
            .attr('r', 0)
            .remove();

        const enterNodes = nodeSelection.enter()
            .append('g')
            .attr('class', 'agent-node')
            .attr('cursor', 'pointer')
            .on('mouseover', (event, d) => this.showTooltip(event, d))
            .on('mouseout', () => this.hideTooltip())
            .on('click', (event, d) => this.onAgentClick(event, d));

        // Agent circle
        enterNodes.append('circle')
            .attr('r', 0)
            .transition()
            .duration(300)
            .attr('r', 18);

        // Agent icon
        enterNodes.append('text')
            .attr('class', 'agent-icon')
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'central')
            .attr('font-size', '14px')
            .attr('pointer-events', 'none');

        // Merge and update
        const allNodes = enterNodes.merge(nodeSelection);

        // Only animate if position actually changed
        allNodes.each(function(d) {
            const node = d3.select(this);
            const currentTransform = node.attr('transform');
            const newTransform = `translate(${d.x}, ${d.y})`;

            // Skip transition if position hasn't changed significantly
            if (currentTransform && currentTransform !== newTransform) {
                node.transition()
                    .duration(300)
                    .attr('transform', newTransform);
            } else if (!currentTransform) {
                node.attr('transform', newTransform);
            }
        });

        allNodes.select('circle')
            .attr('fill', d => this.getAgentColor(d))
            .attr('stroke', d => this.getAgentStroke(d))
            .attr('stroke-width', 2)
            .attr('opacity', d => d.agent.is_arrested ? 0.4 : 1);

        allNodes.select('text')
            .text(d => this.getAgentIcon(d));
    }

    getAgentColor(d) {
        if (d.agent.is_arrested) return '#4a4a4a';
        if (d.agent.is_recruited) return '#065f46';

        const colors = {
            'farmer': '#3d5a1f',
            'guard': '#1e40af',
            'rebel': '#991b1b',
        };
        return colors[d.role] || '#4a4a6a';
    }

    getAgentStroke(d) {
        if (d.agent.is_recruited) return '#10b981';
        if (d.agent.true_faction === 'rebel' && d.role !== 'rebel') {
            return '#ef4444'; // Secret rebel indicator
        }
        return '#6a6a8a';
    }

    getAgentIcon(d) {
        if (d.agent.is_arrested) return '🔒';
        const icons = {
            'farmer': '🌾',
            'guard': '🛡️',
            'rebel': '⚔️',
        };
        return icons[d.role] || '👤';
    }

    showTooltip(event, d) {
        const agent = d.agent;
        const html = `
            <div class="tooltip-header">
                <strong>${agent.name}</strong>
                <span class="tooltip-role">${agent.role}</span>
            </div>
            <div class="tooltip-stats">
                <div>Energy: ${agent.energy}</div>
                <div>Gold: ${agent.gold}</div>
                <div>Suspicion: ${(agent.suspicion * 100).toFixed(0)}%</div>
            </div>
            ${agent.is_recruited ? '<div class="tooltip-tag recruited">Recruited</div>' : ''}
            ${agent.is_arrested ? '<div class="tooltip-tag arrested">Arrested</div>' : ''}
            ${agent.true_faction === 'rebel' && agent.role !== 'rebel' ? '<div class="tooltip-tag secret">Secret Rebel</div>' : ''}
        `;

        this.tooltip.innerHTML = html;
        this.tooltip.style.display = 'block';
        this.tooltip.style.left = `${event.pageX + 10}px`;
        this.tooltip.style.top = `${event.pageY - 10}px`;
    }

    hideTooltip() {
        this.tooltip.style.display = 'none';
    }

    updateNodePositions() {
        // Update road positions
        const roadData = this.roads.map(([from, to]) => {
            const fromLoc = this.locations.find(l => l.id === from);
            const toLoc = this.locations.find(l => l.id === to);
            return { from: fromLoc, to: toLoc };
        });

        this.roadGroup.selectAll('line')
            .data(roadData)
            .attr('x1', d => d.from.x * this.width)
            .attr('y1', d => d.from.y * this.height)
            .attr('x2', d => d.to.x * this.width)
            .attr('y2', d => d.to.y * this.height);

        // Update location positions
        this.nodeGroup.selectAll('.location-group')
            .data(this.locations)
            .attr('transform', d => `translate(${d.x * this.width}, ${d.y * this.height})`);

        // Update location labels
        this.labelGroup.selectAll('.location-label')
            .data(this.locations)
            .attr('x', d => d.x * this.width)
            .attr('y', d => d.y * this.height + 50);

        // Update home positions
        this.updateHomeNodes();

        // Update node fixed positions
        this.nodes.forEach(node => {
            if (node.type === 'location') {
                const loc = this.locations.find(l => l.id === node.id);
                if (loc) {
                    node.fx = loc.x * this.width;
                    node.fy = loc.y * this.height;
                }
            } else if (node.type === 'home') {
                const home = this.agentHomes[node.agentId];
                if (home) {
                    node.fx = home.x * this.width;
                    node.fy = home.y * this.height;
                }
            }
        });
    }

    // Handle click on agent node
    onAgentClick(event, d) {
        event.stopPropagation();
        event.preventDefault();

        // Hide tooltip to prevent visual glitches
        this.hideTooltip();

        // Dispatch custom event that app.js will listen for
        const customEvent = new CustomEvent('agentSelected', {
            detail: { agentId: d.id, agent: d.agent }
        });
        window.dispatchEvent(customEvent);
    }

    // Highlight a conversation between two agents
    highlightConversation(agent1Id, agent2Id, duration = 2000) {
        const node1 = this.nodes.find(n => n.id === agent1Id);
        const node2 = this.nodes.find(n => n.id === agent2Id);

        if (!node1 || !node2) return;

        // Draw a conversation line
        const convLine = this.linkGroup.append('line')
            .attr('class', 'conversation-line')
            .attr('x1', node1.x)
            .attr('y1', node1.y)
            .attr('x2', node2.x)
            .attr('y2', node2.y)
            .attr('stroke', '#f59e0b')
            .attr('stroke-width', 3)
            .attr('stroke-opacity', 0)
            .attr('stroke-dasharray', '5,3');

        convLine.transition()
            .duration(300)
            .attr('stroke-opacity', 1)
            .transition()
            .duration(duration)
            .attr('stroke-opacity', 0)
            .remove();
    }
}

// Export TownGraph class for use in app.js
// Graph initialization is handled in app.js to avoid duplication
window.TownGraph = TownGraph;
