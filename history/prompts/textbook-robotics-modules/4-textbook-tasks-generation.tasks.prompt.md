---
id: 4
title: textbook-tasks-generation
stage: tasks
date: 2025-12-09
surface: agent
model: Qwen
feature: textbook-robotics-modules
branch: 1-textbook-robotics-modules
user: user
command: /sp.tasks
labels: [textbook, robotics, tasks, docusaurus]
links:
  spec: specs/1-textbook-robotics-modules/spec.md
  plan: specs/1-textbook-robotics-modules/plan.md
  adr: null
  pr: null
files:
 - specs/1-textbook-robotics-modules/tasks.md
tests:
 - none
---

## Prompt

```
/sp.tasks Context for task generation: <!-- TASKS GENERATION PROMPT FOR SPEC-KIT PLUS This prompt instructs the agent to generate a complete /sp.tasks file for the "Physical AI & Humanoid Robotics" textbook project. Requirements: - Follow the Constitution, Specs, and Plan strictly. - All tasks must be actionable, atomic, and testable. - Group tasks by phases. - No technical implementation details beyond what is required for textbook creation. --> # Tasks: Physical AI & Humanoid Robotics Textbook ## Task Generation Rules You must generate all tasks required to execute the Implementation Plan and fulfill the Specs while remaining fully compliant with the Constitution. Tasks must be: - **Atomic**: one task = one clear action. - **Non-overlapping**: no duplicated or redundant tasks. - **Phase-aligned**: follow phases defined in /sp.plan. - **Testable**: each task must have measurable acceptance criteria. - **Organized**: grouped under Phase 0, Phase 1, Phase 2, Phase 3. - **Book-focused**: tasks should involve writing chapters, content, diagrams, labs, MCQs, references, and structure — NOT environment setup or ROS/Unity installation. Each task must include: - **id** (unique, snake_case) - **title** - **description** - **phase** (0–3) - **dependencies** (array, can be empty) - **acceptance_criteria** (clear and testable) - **labels** (choose from: writing, research, diagrams, labs, mcqs, editing, structure, qa) ## Phase Definitions ### Phase 0 — Research & Learning Outcomes Tasks should include: - Researching learning outcomes for each module - Gathering topic references - Drafting Gherkin patterns - Collecting real-world robotics examples - Setting theory depth per chapter ### Phase 1 — Structure, Templates & Contracts Tasks should include: - Creating chapter templates - Creating module folders - Defining section/subsection skeletons - Setting diagram placeholders - Drafting lab templates - Drafting MCQ templates ### Phase 2 — Full Chapter Writing Tasks should include writing **all 27 chapters**, each with: - 6–8 learning outcomes - 5–7 Gherkin specs - Theory section - Mermaid diagrams - 2–3 labs - Sim-to-real notes - 12–15 MCQs - References Each chapter must be its own task or multi-task cluster. ### Phase 3 — QA, Polish & Accessibility Tasks should include: - Content QA - Accuracy checks - Diagram verification - Lab execution review (conceptual) - MCQ correctness checks - Urdu translation layer checks - Docusaurus formatting consistency - RAG chunking validation (1024/200) ## Output Format Produce output strictly in **pure YAML**, no Markdown formatting. Top-level key must be: `tasks:` Each task must follow this schema: ```yaml - id: unique_task_id title: "Task title" description: "What must be done" phase: X dependencies: [] acceptance_criteria: - "criterion 1" - "criterion 2" labels: [writing, diagrams] ``` ## Important Restrictions - **Do NOT** include implementation tasks about ROS, Unity, Isaac, or installations. - **Do NOT** generate code. - **Do NOT** add tasks that modify build systems. - **ONLY** generate tasks for writing and structuring the textbook. --- # Final Instruction Generate **50–120 high-quality tasks** following all rules above.
```

## Response snapshot

Created comprehensive tasks file for the Physical AI & Humanoid Robotics textbook with 270 tasks organized into 10 phases. Tasks follow the checklist format with proper IDs, user story labels (US1-US6), and file paths. Tasks span from initial research to final QA, focusing on content creation for all 4 modules and 27 chapters, plus navigation, personalization, AI chatbot integration, multilingual support, and quality assurance.

## Outcome

- ✅ Impact: Complete task breakdown for textbook creation with all required components
- 🧪 Tests: none
- 📁 Files: Created tasks.md with 270 tasks organized by user stories
- 🔁 Next prompts: Begin implementation of individual tasks
- 🧠 Reflection: Tasks comprehensively cover all requirements from constitution and spec

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): not applicable
- Prompt variant (if applicable): not applicable
- Next experiment (smallest change to try): begin with Phase 0 research tasks