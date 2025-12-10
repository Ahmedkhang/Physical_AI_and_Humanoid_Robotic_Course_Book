---
id: 1
title: update-project-constitution
stage: constitution
date: 2025-12-09
surface: agent
model: Qwen
feature: none
branch: 002-docusaurus-frontend-spec
user: user
command: /sp.constitution
labels: [constitution, textbook, robotics]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

```
/sp.constitution You are an expert Spec-Kit Plus engineer. Generate ONLY a complete, valid /sp.constitution Markdown-compatible file for a Docusaurus textbook titled \"Physical AI & Humanoid Robotics\".

Follow these instructions strictly:

Book Title: Physical AI & Humanoid Robotics
Subtitle: Embodied Intelligence with ROS 2, NVIDIA Isaac Sim & Vision-Language-Action Models
Modules and Chapters:

Module 1 – The Robotic Nervous System (ROS 2) → 7 chapters

Module 2 – The Digital Twin (Gazebo & Unity) → 6 chapters

Module 3 – The AI-Robot Brain (NVIDIA Isaac) → 8 chapters

Module 4 – Vision-Language-Action (VLA) → 6 chapters

Chapter Requirements (for each chapter):

6–8 learning outcomes

5–7 Gherkin specs

Theory and core concepts with Mermaid diagrams

2–3 hands-on exercises (examples or labs as content only, not executable code)

Sim-to-real notes written as explanatory content, not instructions to install or run hardware

12–15 high-quality MCQs

Further reading section

RAG Settings: chunk_size 1024, overlap 200
Enable Bonuses: personalization, urdu_translation, background_survey, claude_subagents
Theme: Panaversity

Do not include: instructions to install ROS 2, NVIDIA Isaac, Unity, or any hardware/software setup. Focus purely on textbook content.

Output only the raw constitution Markdown that can be saved directly as /sp.constitution for Spec-Kit Plus in a Docusaurus project. Do not add explanations, code fences, or extra words.
```

## Response snapshot

Created a complete constitution file for the "Physical AI & Humanoid Robotics" textbook with six core principles, book structure details, development workflow, and governance sections. The constitution defines requirements for each of the four modules and their chapters, emphasizing educational excellence, practical application, progressive skill building, real-world relevance, technical accuracy, accessibility, and inclusion.

## Outcome

- ✅ Impact: Updated project constitution with specific requirements for the robotics textbook
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md
- 🔁 Next prompts: Module content generation
- 🧠 Reflection: Constitution now properly defines the structure and principles for the textbook project

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): not applicable
- Prompt variant (if applicable): not applicable
- Next experiment (smallest change to try): begin generating textbook content per constitution