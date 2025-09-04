import React, { useState, useEffect } from 'react';
import './DataExplorationPage.css';

const DataExplorationPage = () => {
    const [graphData, setGraphData] = useState(null);
    const [selectedNode, setSelectedNode] = useState(null);

    // This is a placeholder for fetching data from the API.
    useEffect(() => {
        const fetchGraphData = async () => {
            // TODO: Call API to fetch graph data from the CKG
            const mockData = {
                nodes: [
                    { id: 'Agent', label: 'Agent', color: '#1abc9c' },
                    { id: 'Action', label: 'Action', color: '#3498db' },
                    { id: 'Reason', label: 'Reason', color: '#9b59b6' },
                    { id: 'move_1', label: 'Move 1', color: '#e74c3c' },
                    { id: 'strategic_goal', label: 'Strategic Goal', color: '#f1c40f' }
                ],
                links: [
                    { source: 'Agent', target: 'Action' },
                    { source: 'Action', target: 'move_1' },
                    { source: 'move_1', target: 'strategic_goal' }
                ]
            };
            setGraphData(mockData);
        };
        fetchGraphData();
    }, []);

    const handleNodeClick = (node) => {
        setSelectedNode(node);
        // TODO: Fetch detailed node information from CKG API
        console.log(`Node selected: ${node.id}`);
    };

    return (
        <div className="data-exploration-page">
            <header className="exploration-header">
                <h1>Data & Knowledge Exploration</h1>
                <p>Visualize the AI's internal Conceptual Knowledge Graph and verify its decisions.</p>
            </header>
            <main className="exploration-content">
                <div className="graph-container">
                    {/* This would be where a D3.js or other graph visualization library would render the graph */}
                    <div className="mock-graph">
                        {graphData ? (
                            "Graph visualization of the CKG would go here."
                        ) : (
                            "Loading graph data..."
                        )}
                    </div>
                </div>
                <div className="sidebar">
                    <h3>Node Details</h3>
                    {selectedNode ? (
                        <div className="node-details">
                            <p><strong>ID:</strong> {selectedNode.id}</p>
                            <p><strong>Label:</strong> {selectedNode.label}</p>
                            {/* TODO: Add more details from CKG query, including verifiability score */}
                            <p><strong>Type:</strong> Placeholder</p>
                            <p><strong>Description:</strong> Placeholder</p>
                            <button className="verify-button">Verify on Blockchain</button>
                        </div>
                    ) : (
                        <p>Click on a node to see its details.</p>
                    )}
                </div>
            </main>
        </div>
    );
};

export default DataExplorationPage;

