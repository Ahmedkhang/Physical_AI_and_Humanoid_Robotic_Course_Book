# Data Model: Physical AI & Humanoid Robotics Textbook

## Module
- **Fields**: 
  - id (string, required): Unique identifier for the module
  - title (string, required): Display title of the module
  - description (string, required): Brief description of module contents
  - weekRange (string, required): Time span for module completion (e.g. "Weeks 3-5")
  - chapterCount (integer, required): Number of chapters in the module
  - sortOrder (integer, required): Order in which modules appear
- **Relationships**: Contains many Chapter entities
- **Validation**: 
  - title must be 3-100 characters
  - weekRange must follow format "Weeks X-Y"
  - chapterCount must be positive integer
  - sortOrder must be positive integer

## Chapter
- **Fields**:
  - id (string, required): Unique identifier for the chapter
  - title (string, required): Display title of the chapter
  - moduleId (string, required): Reference to parent Module
  - week (integer, required): Week in which chapter is studied
  - difficulty (string, required): One of "beginner", "intermediate", "advanced"
  - estimatedHours (integer, required): Estimated time to complete chapter
  - sortOrder (integer, required): Order of chapter within module
  - learningOutcomes (array of strings, required): 6-8 measurable outcomes
  - gherkinSpecs (array of strings, required): 5-7 Gherkin scenarios
  - theorySection (string, required): Content of theory & intuition section
  - coreConcepts (object, required): Core concepts with diagrams and tables
  - handsOnLabs (array of objects, required): 2-3 hands-on labs
  - simToRealNotes (string, optional): Notes for real-world application
  - mcqs (array of objects, required): 12-15 multiple choice questions
  - furtherReading (array of objects, required): 4-6 references
- **Relationships**: 
  - Belongs to one Module
  - Contains many Section entities
- **Validation**:
  - title must be 3-100 characters
  - moduleId must reference valid Module
  - difficulty must be one of specified values
  - estimatedHours must be positive number between 1-10
  - learningOutcomes must have 6-8 items
  - gherkinSpecs must have 5-7 items
  - handsOnLabs must have 2-3 items
  - mcqs must have 12-15 items

## Section
- **Fields**:
  - id (string, required): Unique identifier for the section
  - title (string, required): Display title of the section
  - chapterId (string, required): Reference to parent Chapter
  - sortOrder (integer, required): Order in which section appears in chapter
- **Relationships**:
  - Belongs to one Chapter
  - Contains many Subsection entities
- **Validation**:
  - title must be 3-50 characters
  - chapterId must reference valid Chapter
  - sortOrder must be positive integer

## Subsection
- **Fields**:
  - id (string, required): Unique identifier for the subsection
  - title (string, required): Display title of the subsection
  - sectionId (string, required): Reference to parent Section
  - sortOrder (integer, required): Order in which subsection appears in section
  - topics (array of strings, required): Bullet points of topics covered
- **Relationships**:
  - Belongs to one Section
  - Contains many Content entities
- **Validation**:
  - title must be 3-50 characters
  - sectionId must reference valid Section
  - topics must have 3-15 items

## Content
- **Fields**:
  - id (string, required): Unique identifier for content item
  - type (string, required): Type of content - one of "theory", "lab", "concept", "mcq", "reference"
  - title (string, required): Title of the content
  - contentData (string, required): Actual content in markdown format
  - parentId (string, required): Reference to parent Subsection
  - parentType (string, required): Type of parent entity
  - sortOrder (integer, required): Order within parent
- **Validation**:
  - type must be one of specified values
  - title must be 3-100 characters
  - contentData must be non-empty
  - sortOrder must be positive integer

## UserProfile
- **Fields**:
  - id (string, required): Unique identifier for the user
  - languagePreference (string, required): User's preferred language (e.g. "English", "Urdu")
  - learningPace (string, required): One of "slow", "moderate", "fast"
  - backgroundSurveyResults (object, required): Results from background survey
  - completedChapters (array of strings, required): IDs of chapters user has completed
  - currentChapter (string, optional): ID of chapter user is currently studying
- **Validation**:
  - languagePreference must be supported language
  - learningPace must be one of specified values
  - completedChapters must contain valid chapter IDs
  - currentChapter must be valid chapter ID if specified

## CoreConcepts
- **Fields**:
  - id (string, required): Unique identifier for the core concepts
  - chapterId (string, required): Reference to parent Chapter
  - mermaidDiagrams (array of strings, required): Mermaid diagram definitions
  - tables (array of objects, required): Data tables with headers and rows
- **Relationships**: Belongs to one Chapter
- **Validation**:
  - mermaidDiagrams must contain valid Mermaid syntax
  - tables must have headers and rows properties

## Lab
- **Fields**:
  - id (string, required): Unique identifier for the lab
  - title (string, required): Display title of the lab
  - chapterId (string, required): Reference to parent Chapter
  - description (string, required): Purpose and objective of the lab
  - steps (array of strings, required): Step-by-step instructions
  - codeExamples (array of objects, required): Code snippets with language identifiers
  - expectedOutcome (string, required): What the lab should produce
- **Relationships**: Belongs to one Chapter
- **Validation**:
  - steps must have 5-20 items
  - codeExamples must include language and code properties

## MCQ
- **Fields**:
  - id (string, required): Unique identifier for the question
  - question (string, required): The question text
  - chapterId (string, required): Reference to parent Chapter
  - options (array of strings, required): Possible answer choices
  - correctAnswer (string, required): The correct option
  - explanation (string, required): Explanation of why the answer is correct
- **Relationships**: Belongs to one Chapter
- **Validation**:
  - options must have 3-6 items
  - correctAnswer must match one of the options
  - explanation must be 10-200 characters

## Reference
- **Fields**:
  - id (string, required): Unique identifier for the reference
  - title (string, required): Title of the referenced material
  - url (string, required): URL to the resource
  - chapterId (string, optional): Reference to associated Chapter (if applicable)
  - description (string, optional): Brief description of the resource
- **Validation**:
  - url must be valid URL format
  - title must be 3-100 characters