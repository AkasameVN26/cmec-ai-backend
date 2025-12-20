# CMEC AI Backend System Design

This document outlines the system architecture and data flow for the CMEC AI Backend.

## 1. System Architecture

The following flowchart illustrates the high-level architecture and component interactions.

```mermaid
graph TD
    Client[Client Application]

    subgraph "FastAPI Backend"
        API[API Endpoints]
        subgraph "Services"
            SupaSvc[Supabase Service]
            AISvc[AI Service]
            MetSvc[Metric Service]
        end
    end

    subgraph "External Systems"
        Supabase[(Supabase Database)]
        Ollama[Ollama AI Model]
    end

    %% Client Interactions
    Client -->|HTTP Requests| API

    %% API to Services
    API -->|Get Data| SupaSvc
    API -->|Generate Summary| AISvc
    API -->|Explain/Evaluate| MetSvc

    %% Service Interactions
    SupaSvc <-->|Query/Fetch| Supabase
    AISvc <-->|Chat Completion| Ollama
    AISvc -.->|Calculate Metrics (Optional)| MetSvc

    %% Service Details
    SupaSvc -- Aggregates Patient Data --> API
    MetSvc -- Embeddings & ROUGE --> API
```

### Components Description

*   **Client Application**: The frontend or external system initiating requests.
*   **API Endpoints (`app/api/endpoints.py`)**: The entry point for HTTP requests. Handles routing and response formatting.
*   **Supabase Service (`app/services/supabase_service.py`)**: Responsible for fetching and aggregating patient data from Supabase. It performs parallel queries to fetch patient info, allergies, visits, diagnoses, notes, lab results, and prescriptions, combining them into a structured prompt.
*   **AI Service (`app/services/ai_service.py`)**: Interacts with the Ollama API to generate medical summaries. It constructs the prompt using the aggregated data and system instructions.
*   **Metric Service (`app/services/metric_service.py`)**: Handles the "Explainability" and evaluation metrics. It uses local models (SentenceTransformer, ROUGE) to calculate similarity scores and map summary sentences back to source data.
*   **Supabase**: The external Postgres database hosting medical records.
*   **Ollama**: The external AI inference engine running the LLM.

---

## 2. Data Flow: Summarization

This sequence diagram details the process of generating a medical summary (`/api/summarize/{id}`).

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant SupaSvc as SupabaseService
    participant DB as Supabase
    participant AISvc as AIService
    participant Ollama
    participant MetSvc as MetricService

    Client->>API: GET /summarize/{id}

    rect rgb(240, 248, 255)
    note right of API: Step 1: Data Preparation
    API->>SupaSvc: prepare_input(id)
    par Fetch Independent Data
        SupaSvc->>DB: Fetch Patient Info
        SupaSvc->>DB: Fetch Allergies
        SupaSvc->>DB: Fetch Visits
        SupaSvc->>DB: Fetch Diagnoses
        SupaSvc->>DB: Fetch Notes
    and
        DB-->>SupaSvc: Return Data
    end

    opt If Visits Exist
        SupaSvc->>DB: Fetch Lab Results
        SupaSvc->>DB: Fetch Prescriptions
        DB-->>SupaSvc: Return Data
    end

    SupaSvc->>SupaSvc: Aggregate & Format Prompt
    SupaSvc-->>API: PreparedInput (Prompt Content)
    end

    rect rgb(255, 250, 240)
    note right of API: Step 2: AI Generation
    API->>AISvc: summarize(prompt)
    AISvc->>Ollama: POST /api/chat (Prompt + System Instr)
    Ollama-->>AISvc: Generated Summary Text
    end

    opt If measure_metrics=True
        rect rgb(240, 255, 240)
        AISvc->>MetSvc: calculate_metrics(source, summary)
        MetSvc->>MetSvc: Tokenize & Embed
        MetSvc->>MetSvc: Calculate ROUGE & Cosine Sim
        MetSvc-->>AISvc: Metrics Data
        end
    end

    AISvc-->>API: SummaryResponse
    API-->>Client: JSON Response
```

---

## 3. Data Flow: Explanation

This sequence diagram details the process of explaining a summary (`/api/explain/{id}`).

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant SupaSvc as SupabaseService
    participant DB as Supabase
    participant MetSvc as MetricService

    Client->>API: POST /explain/{id} (payload: summary text)

    note right of API: Step 1: Retrieve Source Data
    API->>SupaSvc: prepare_input(id)
    SupaSvc->>DB: Fetch All Patient Data (Parallel)
    DB-->>SupaSvc: Data
    SupaSvc-->>API: PreparedInput (Source Segments)

    note right of API: Step 2: Compute Explanation
    API->>MetSvc: explain_summary(source_segments, summary)

    MetSvc->>MetSvc: Segment Summary into Sentences
    MetSvc->>MetSvc: Segment Source Notes into Sentences

    note right of MetSvc: Embedding & Similarity
    MetSvc->>MetSvc: Generate Embeddings (SentenceTransformer)
    MetSvc->>MetSvc: Calculate Cosine Similarity Matrix

    loop For each summary sentence
        MetSvc->>MetSvc: Find best matching source sentence
        MetSvc->>MetSvc: Identify context neighbors
    end

    MetSvc-->>API: Explanation (Matches & Scores)
    API-->>Client: ExplainResponse
```
