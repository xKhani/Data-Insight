Product Requirements Document (PRD): Data Insight
1. Problem Statement
The Bottleneck: Extracting actionable insights from raw data currently requires high technical proficiency in Python, statistics, and data storytelling. For researchers and students, the process of performing Exploratory Data Analysis (EDA), generating visualizations, and manually drafting a scientific report is repetitive and fragmented. A single LLM prompt often fails to handle large datasets accurately or produce high-quality visual artifacts without hallucinating code.

The Solution: Data Insight is an agentic framework that automates the transition from a raw CSV file to a formatted scientific report. It replaces manual scripting with an autonomous multi-agent system that reasons through the data, creates visual evidence, and synthesizes findings through a structured, stateful workflow.

2. User Personas
Academic Researcher: Needs to quickly validate hypotheses on new datasets and generate formal draft reports for further refinement.

Junior Data Analyst: Seeks to automate the "boilerplate" parts of EDA (cleaning, basic plotting, and summary statistics) to focus on high-level strategy.

Student: Uses the tool to observe how agents logically structure a data inquiry, helping them learn standard data science workflows.

3. Success Metrics
Lead Time Reduction: Decrease the time taken to move from "raw CSV" to "first draft report" by at least 70% compared to manual coding.

Artifact Completeness: Successfully produce a three-part output (Statistical Summary, Visual Plots, and Narrative Report) in 95% of successful runs.

Agent Autonomy: The system must perform at least five distinct logical steps (Hypothesis → Code → Plot → Review → Report) from a single initial user prompt without crashing.

4. Tool & Data Inventory (The External World)
To satisfy the agentic boundary of Perceive, Reason, and Execute, the system interacts with the following:

Knowledge Sources: * User Datasets: Structured CSV files provided at runtime.

Research Papers: Integration with ArXiv or statistical Wikis to ground analysis in scientific context.

Action Tools:

Python_REPL: A sandboxed environment for executing analysis and visualization code.

Matplotlib / Seaborn: Libraries for automated, dynamic chart generation.

File_System_API: For saving, retrieving, and organizing "artifacts" (PNGs, Markdown files).

Web_Search: For fetching external references to support the generated hypothesis.

5. System Architecture (LangGraph Framework)
The project is built using LangGraph to implement a stateful, cyclic multi-agent system.

5.1 State Management (GraphState)
The system uses a centralized GraphState object to track progress. This shared memory ensures that context—such as dataset schemas and previous error logs—is preserved as the control passes between nodes.

5.2 Nodes and Edge Logic
Nodes (Agents): Each agent (Hypothesis, Coder, Viz, Reviewer) acts as a node, performing a specific transformation on the state.

The Supervisor (Orchestrator): Uses conditional logic to route the workflow. For example, if the Quality Review Agent identifies a bug in the code, a conditional edge routes the state back to the Coder Agent for self-correction rather than proceeding to the report stage.

5.3 Agentic Loop & Self-Correction
This implementation utilizes a Review-and-Revise loop. If the Python_REPL returns an error (e.g., a KeyError), the error message is fed back into the state. The agents "reason" over the error log and regenerate the script, ensuring the system is resilient and autonomous.

5.4 Human-in-the-Loop
Using LangGraph’s breakpoint feature, the system pauses after the Hypothesis and Visualization stages. This allows the user to approve the trajectory or provide feedback before the Report Agent finalizes the document.

6. Conclusion
By utilizing a LangGraph-based architecture, Data Insight moves beyond a simple "chatbot" into a robust industrial agent. It effectively manages complex dependencies between data analysis and narrative writing, delivering a reliable tool for automated scientific reporting.
