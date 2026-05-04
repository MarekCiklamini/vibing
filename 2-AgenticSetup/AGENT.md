Input document
→ detect headings
→ split into sections
→ split sections into paragraphs
→ group paragraphs into semantic chunks
→ store metadata

# AGENTS.md
## Semantic Document Intelligence System (Chunking → Ontology → Retrieval)

---

## 1. Purpose

This system transforms raw documents into multi-layer semantic representations enabling:

- High-quality retrieval (RAG)
- Knowledge graph construction
- Agent-based reasoning
- Scientific and technical document understanding

Core principle:

Preserve meaning across hierarchical levels, not token boundaries.

---

## 2. System Overview

Documents  
→ Parser  
→ Hierarchical Chunker  
→ Semantic Enrichment  
→ Storage Layer  
    - Vector DB (embeddings)  
    - Graph (ontology)  
    - Metadata store  
→ Retrieval Layer  
→ Agent Interface  

---

## 3. Agents

### 3.1 Parser Agent

Role:
Convert raw documents into structured format.

Inputs:
- PDF / Markdown / TXT

Outputs:
- Document with sections, subsections, paragraphs

Responsibilities:
- Detect headings
- Extract structure
- Preserve paragraph boundaries
- Maintain order

---

### 3.2 Chunking Agent

Role:
Create hierarchical semantic chunks.

Levels:
- Section
- Paragraph group (topic block)
- Paragraph

Rules:
- Do not split inside a paragraph
- Prefer semantic boundaries over token limits
- Maintain parent-child relationships

---

### 3.3 Semantic Enrichment Agent

Role:
Extract meaning from chunks.

Extract:
- Entities
- Concepts
- Claims
- Relations
- Semantic role

Semantic roles include:
- definition
- method
- evidence
- result
- limitation
- conclusion

---

### 3.4 Embedding Agent

Role:
Convert chunks into vector embeddings.

Responsibilities:
- Generate embeddings for chunk text
- Optionally embed summaries
- Store in vector database

---

### 3.5 Ontology Agent

Role:
Build knowledge graph from chunks.

Graph relationships:
- Chunk → mentions → Entity
- Entity → relates_to → Entity
- Chunk → supports → Claim
- Chunk → belongs_to → Section

Responsibilities:
- Normalize entities
- Merge duplicates
- Maintain graph consistency

---

### 3.6 Storage Agent

Role:
Persist all artifacts.

Stores:
- Chunk data (JSON or DB)
- Embeddings (vector DB)
- Ontology graph

Each chunk should include:
- text
- level
- section
- entities
- claims
- relations
- semantic role
- parent and children references

---

### 3.7 Retrieval Agent

Role:
Hybrid retrieval across multiple layers.

Steps:
1. Embed query
2. Vector search (top-k chunks)
3. Expand context:
   - parent section
   - neighboring chunks
4. Traverse ontology:
   - entity matches
   - related claims
5. Rerank results

Output:
- Ranked, context-rich chunks

---

### 3.8 Reasoning Agent (optional)

Role:
Operate on retrieved knowledge.

Capabilities:
- Question answering
- Summarization
- Structured output generation
- Multi-hop reasoning

---

## 4. Data Model

Each chunk should contain:

- chunk_id
- document_id
- level (section / paragraph_group / paragraph)
- text
- summary
- section
- order
- parent_id
- children_ids
- entities
- claims
- relations
- semantic_role
- embedding

---

## 5. Pipeline Execution

1. Parser Agent  
2. Chunking Agent  
3. Semantic Enrichment Agent  
4. Embedding Agent  
5. Ontology Agent  
6. Storage Agent  

---

## 6. Key Design Principles

### Hierarchical integrity
- Preserve document structure
- Avoid flattening

### Semantic coherence
- Each chunk must represent a meaningful unit
- Avoid splitting:
  - claim and evidence
  - method and result

### Multi-resolution access
Support navigation across:
- document → section → chunk → claim → entity

### Hybrid retrieval
Combine:
- vector similarity
- metadata filtering
- ontology traversal
- hierarchical expansion

### Modularity
- Agents must be replaceable
- Components independently testable
- Ready for orchestration frameworks

---

## 7. Evaluation Strategy

Chunking quality:
- semantic coherence
- boundary correctness

Retrieval quality:
- precision@k
- recall@k

Answer quality:
- groundedness
- citation accuracy

---

## 8. MVP Scope

Start with:
- Section and paragraph chunking
- Embeddings and vector search
- Basic metadata

Then extend:
- Entity extraction
- Claim extraction
- Ontology graph
- Hybrid retrieval

---

## 9. Future Extensions

- Graph embeddings (GNN)
- Sequence modeling over chunks
- Multi-document reasoning
- Agent orchestration
- Real-time ingestion
- Domain-specific ontologies

---

## 10. Suggested Stack

- Python
- LangGraph (optional)
- Qdrant (vector DB)
- NetworkX (graph)
- FastAPI (API layer)

---

## Final Guiding Principle

Paragraphs are for reading.  
Chunks are for retrieval.  
Claims are for reasoning.  
Graphs are for intelligence.
