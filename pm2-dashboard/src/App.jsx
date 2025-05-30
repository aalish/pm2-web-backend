import React from "react";
// import PM2Dashboard from './PM2Dashboard';

import "./App.css";
import CollapsibleTable from "./dashboard";

function App() {
  return (
    <div>
      <h2>PM2 Process Dashboard</h2>
      <CollapsibleTable />
    </div>
  );
}

export default App;
