// zenith_ui/src/App.js

import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import HomePage from './pages/HomePage';
import DashboardPage from './pages/DashboardPage';
import DataExplorationPage from './pages/DataExplorationPage';
import SettingsPage from './pages/SettingsPage';
import './App.css'; // You'll need to create this for styling

const App = () => {
    return (
        <Router>
            <div className="app-container">
                <nav className="main-nav">
                    <Link to="/" className="nav-link">Home</Link>
                    <Link to="/dashboard" className="nav-link">Dashboard</Link>
                    <Link to="/data" className="nav-link">Data</Link>
                    <Link to="/settings" className="nav-link">Settings</Link>
                </nav>
                <div className="content-container">
                    <Routes>
                        <Route path="/" element={<HomePage />} />
                        <Route path="/dashboard" element={<DashboardPage />} />
                        <Route path="/data" element={<DataExplorationPage />} />
                        <Route path="/settings" element={<SettingsPage />} />
                    </Routes>
                </div>
            </div>
        </Router>
    );
};

export default App;
