# Context Engine: Technical Architecture Documentation

## What is a Context Engine?

A context engine is a retrieval-augmented system that transforms disparate information sources into a unified, searchable knowledge base. Unlike traditional search engines that match keywords, a context engine understands semantic relationships between concepts and preserves the contextual metadata that makes information actionable.

The core challenge lies in the heterogeneity problem: emails contain temporal context and conversation threads, while documentation contains hierarchical structure and cross-references. A context engine must normalize these different data types into a uniform representation while preserving their unique contextual properties.

Consider the difference between searching for "authentication implementation" in a traditional system versus a context engine. Traditional search returns documents containing those words. A context engine returns the email thread where you discussed OAuth2 integration challenges, the Notion page documenting your API design decisions, and the code snippet from a colleague's implementation—all ranked by contextual relevance to your current project.

## Some results from this research

![carbon (6).png](https://i.imgur.com/y8U6ofO.jpeg)

![screencapture-localhost-3000-2025-06-16-11_03_49.png](https://i.imgur.com/cdQMKVt.png)

![screencapture-localhost-3000-2025-06-16-11_04_45.png](https://i.imgur.com/iaCiR7l.png)

## Architecture Overview

The system operates as a multi-stage pipeline that transforms raw information into contextually-aware search results. The architecture consists of five primary components:

**Data Ingestion Layer**: Connects to heterogeneous data sources (Gmail, Notion, file systems) using source-specific protocols (OAuth2, API keys, direct access). Each source requires custom authentication and rate limiting strategies.

**Normalization Engine**: Converts disparate data formats into a unified document model. This layer handles the impedance mismatch between structured data (Notion pages with properties) and unstructured data (email conversations with headers and threading).

**Chunking Pipeline**: Segments large documents into semantically coherent units while preserving cross-chunk relationships. This addresses the fundamental limitation of embedding models, which operate on fixed-length input sequences.

**Vector Embedding System**: Transforms text chunks into high-dimensional numerical representations using pre-trained language models. The system uses sentence-transformers for efficiency, specifically the all-MiniLM-L6-v2 model which balances accuracy and computational overhead.

**Query Processing Engine**: Handles search queries by generating embeddings, performing similarity search, and contextualizing results with metadata. This component manages the translation between user intent and vector space operations.

## Data Ingestion Pipeline

### Multi-Source Authentication Strategy

Different data sources require distinct authentication mechanisms, each with unique security and refresh token handling requirements.

Gmail integration uses OAuth2 with incremental authorization scopes. The system requests minimal permissions (gmail.readonly, calendar.readonly) and implements the desktop application flow rather than web application flow to avoid callback URL complexity. The authentication process stores refresh tokens locally, enabling automated data synchronization without user intervention.

Notion integration relies on API key authentication with workspace-level permissions. Unlike OAuth2, API keys don't expire but provide broader access, requiring careful scope limitation through database-specific integration tokens.

```python
# OAuth2 flow implementation for Gmail
def authenticate_gmail(self):
    flow = InstalledAppFlow.from_client_secrets_file(
        'config/credentials.json',
        scopes=['https://www.googleapis.com/auth/gmail.readonly']
    )
    credentials = flow.run_local_server(port=0)
    return build('gmail', 'v1', credentials=credentials)
```

### Rate Limiting and Batch Processing

Each data source imposes different rate limits that must be respected to maintain system reliability. Gmail API allows 1 billion quota units per day, with most operations consuming 5-10 units. The system implements exponential backoff for 429 responses and batches requests to minimize API calls.

Notion API is more restrictive at 3 requests per second, requiring careful batch sizing and request queuing. The ingestion pipeline implements a producer-consumer pattern where data collection runs at source-appropriate rates while downstream processing operates independently.

### Incremental Synchronization

The system maintains synchronization metadata to enable incremental updates rather than full re-indexing. For Gmail, this means tracking the last processed message ID and using search queries with date ranges. For Notion, it involves monitoring last_edited_time properties and page version numbers.

```python
def get_incremental_emails(self, last_sync_time):
    query = f'after:{last_sync_time.strftime("%Y/%m/%d")}'
    response = self.gmail_service.users().messages().list(
        userId='me', q=query, maxResults=100
    ).execute()
    return response.get('messages', [])
```

## Normalization Techniques

### Unified Document Model

The normalization layer transforms source-specific data structures into a common document representation. This model must accommodate both structured metadata (sender, timestamp, page properties) and unstructured content (email bodies, page content) while preserving relationships between entities.

The unified document schema includes:
- **Content**: The primary text content, stripped of formatting but preserving structure
- **Metadata**: Source-specific properties (email headers, page properties, file metadata)
- **Relationships**: References to other documents, reply chains, page hierarchies
- **Temporal Context**: Creation time, modification time, access patterns
- **Source Attribution**: Origin system, unique identifiers, access permissions

### Content Extraction and Cleaning

Email content requires sophisticated parsing to handle multi-part messages, HTML formatting, and quoted reply chains. The system uses BeautifulSoup for HTML parsing, implementing custom logic to preserve meaningful structure while removing presentation markup.

```python
def extract_email_content(self, email_data):
    if email_data['mimeType'] == 'text/html':
        soup = BeautifulSoup(email_data['body']['data'], 'html.parser')
        # Remove script and style elements
        for element in soup(['script', 'style']):
            element.decompose()
        return soup.get_text(separator=' ', strip=True)
    return email_data['body']['data']
```

Notion pages contain structured content blocks (paragraphs, lists, databases) that must be linearized while preserving semantic relationships. The system handles rich text formatting, embedded links, and cross-page references by maintaining a mapping between content and its structural context.

### Metadata Standardization

Different sources express similar concepts using different schemas. Email "from" addresses become "author" fields, while Notion "created_by" properties serve the same semantic purpose. The normalization layer maps these variations to a consistent vocabulary.

Temporal information requires careful handling due to timezone differences and format variations. The system normalizes all timestamps to UTC and stores both creation and modification times where available.

## Chunking Strategy

### Semantic Segmentation Approach

The chunking pipeline addresses a fundamental constraint of embedding models: they operate on fixed-length input sequences (typically 512 tokens for BERT-family models). Simply truncating documents loses information, while naive sliding windows can split related concepts across chunks.

The system implements a semantic-aware chunking strategy that identifies natural breakpoints in content. For structured documents, it uses markup boundaries (headers, paragraphs, list items). For emails, it preserves quoted reply sections as separate chunks while maintaining thread context.

```python
def chunk_document(self, content, max_chunk_size=400, overlap_size=50):
    sentences = self.sentence_splitter.split(content)
    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_size + sentence_length > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            # Overlap with previous chunk
            current_chunk = current_chunk[-overlap_size:] if len(current_chunk) > overlap_size else current_chunk
            current_size = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_size += sentence_length

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks
```

### Overlap Strategy for Context Preservation

Overlapping chunks ensure that concepts spanning chunk boundaries remain discoverable. The system uses a sliding window approach with 50-token overlap, providing context continuity while minimizing redundancy.

The overlap strategy considers semantic boundaries. Rather than cutting mid-sentence, the system identifies complete thoughts and includes them in adjacent chunks. This approach prevents the loss of coherent ideas that might be split by arbitrary character limits.

### Chunk Metadata Inheritance

Each chunk inherits metadata from its parent document while adding chunk-specific information:
- **Chunk Index**: Position within the document sequence
- **Parent Document ID**: Reference to the original document
- **Content Type**: Text, code, structured data, conversation thread
- **Semantic Markers**: Headings, list items, quoted text indicators

This metadata enables the system to reconstruct document context during result presentation and supports chunk-level relevance scoring.

## Vector Embedding Pipeline

### Model Selection and Trade-offs

The system uses the all-MiniLM-L6-v2 model from sentence-transformers, chosen for its balance between accuracy and computational efficiency. This model produces 384-dimensional embeddings, significantly smaller than larger alternatives like BERT-large (1024 dimensions) while maintaining competitive performance on semantic similarity tasks.

The model selection involved evaluating several factors:
- **Inference Speed**: Critical for real-time query processing
- **Memory Requirements**: Important for deployment on resource-constrained environments
- **Language Coverage**: Support for technical terminology and mixed formal/informal text
- **Domain Adaptation**: Performance on technical documentation and conversational text

### Embedding Generation Process

The embedding pipeline processes chunks in batches to optimize GPU utilization and memory management. The system implements dynamic batching, adjusting batch size based on available resources and input text length.

```python
def generate_embeddings(self, text_chunks):
    embeddings = []
    batch_size = self.calculate_optimal_batch_size(text_chunks)

    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        batch_embeddings = self.model.encode(
            batch,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        embeddings.extend(batch_embeddings.cpu().numpy())

    return embeddings
```

Embedding normalization ensures consistent similarity calculations across different content types. The system applies L2 normalization to all vectors, enabling efficient cosine similarity computation using dot products.

### Vector Storage and Indexing

The system uses Qdrant as the vector database, chosen for its support of filtered searches and metadata-aware querying. Qdrant's hybrid search capabilities allow combining semantic similarity with metadata filtering, enabling queries like "find discussions about authentication from last month."

Vector storage includes both the embedding and comprehensive metadata, enabling complex query patterns:

```python
def store_chunk_vector(self, chunk, embedding):
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding.tolist(),
        payload={
            'content': chunk['content'],
            'source': chunk['source'],
            'metadata': chunk['metadata'],
            'created_at': chunk['timestamp'],
            'chunk_index': chunk['index']
        }
    )
    self.qdrant_client.upsert(collection_name="documents", points=[point])
```

## Context Building Architecture

### Relationship Graph Construction

The system constructs an implicit relationship graph between documents based on multiple signals:
- **Temporal Proximity**: Documents created or accessed within similar timeframes
- **Semantic Similarity**: Content-based relationships detected through embedding similarity
- **Explicit References**: Links, mentions, reply chains, cross-references
- **Collaborative Context**: Shared authorship, participation in conversations

This relationship graph enables context expansion during query processing. When retrieving results for a query, the system can include related documents that provide additional context even if they don't directly match the search terms.

### Contextual Metadata Enrichment

Each document chunk is enriched with contextual signals that improve relevance scoring:

**Temporal Context**: The system tracks when documents were created relative to project milestones, deadlines, or significant events. A technical discussion from last week carries different relevance than one from six months ago.

**Authorship Context**: Documents are tagged with author information and role context. An email from a team lead about architecture decisions carries different weight than a casual conversation.

**Project Context**: The system attempts to infer project associations based on keywords, participant overlap, and temporal clustering. This enables project-scoped searches and context-aware result ranking.

### Context Propagation Strategy

Context propagation ensures that implicit relationships between documents are discoverable through search. The system implements several propagation mechanisms:

**Thread Propagation**: Email conversations inherit context from all messages in the thread. A reply to a technical discussion carries forward the technical context even if the reply itself contains minimal technical content.

**Reference Propagation**: Documents that reference other documents inherit partial context from their targets. A Notion page linking to an email discussion gains relevance for queries related to that discussion topic.

**Collaborative Propagation**: Documents created by the same author or involving the same participants share contextual relationships, enabling discovery of related work across time and projects.

## Query Processing Engine

### Query Understanding and Expansion

The query processing pipeline begins with intent analysis and query expansion. Simple keyword queries are expanded with synonyms and related terms derived from the document corpus. Technical queries benefit from domain-specific expansion (e.g., "auth" expanding to include "authentication," "authorization," "OAuth").

```python
def process_query(self, query_text, context_hint=None):
    # Generate query embedding
    query_embedding = self.encoder.encode(query_text)

    # Apply context-aware filters
    filters = self.build_context_filters(context_hint)

    # Perform semantic search
    search_results = self.qdrant_client.search(
        collection_name="documents",
        query_vector=query_embedding,
        query_filter=filters,
        limit=20,
        with_payload=True
    )

    return self.rank_and_contextualize(search_results, query_text)
```

### Multi-Stage Retrieval Strategy

The system implements a multi-stage retrieval process that balances recall and precision:

**Stage 1 - Broad Retrieval**: Semantic search across all documents using relaxed similarity thresholds. This stage prioritizes recall, retrieving potentially relevant documents even with lower confidence scores.

**Stage 2 - Context Filtering**: Apply metadata filters based on query context (time ranges, source types, authorship). This stage reduces the candidate set while preserving contextual relevance.

**Stage 3 - Relevance Ranking**: Re-rank results using a combination of semantic similarity, recency, and contextual signals. Recent documents receive modest recency boosts, while documents with strong authorship or project context gain additional relevance.

**Stage 4 - Context Expansion**: Include related documents that provide additional context for the top-ranked results. This ensures that users receive comprehensive information rather than isolated fragments.

### Result Contextualization

The final stage transforms raw search results into contextually meaningful responses. This involves:

**Source Attribution**: Clear indication of which system provided each result, enabling users to understand the provenance and reliability of information.

**Relationship Mapping**: Explicit indication of relationships between returned documents (e.g., "This email thread continued the discussion from your Notion page about API design").

**Temporal Context**: Presentation of results within their temporal context, highlighting when information was created or last updated relative to the current query.

**Actionability Enhancement**: Where possible, the system identifies actionable elements in results (links to follow up on, people to contact, decisions that need to be made).

## Performance Considerations and Optimization

### Scalability Architecture

The system architecture supports horizontal scaling at multiple levels:

**Ingestion Scaling**: Data ingestion processes run independently and can be distributed across multiple workers. Each source type can scale independently based on API rate limits and data volume.

**Embedding Generation**: The embedding pipeline supports GPU acceleration and can distribute batch processing across multiple machines. The system implements dynamic load balancing to optimize throughput.

**Vector Search**: Qdrant supports distributed deployment with automatic sharding and replication. The system can handle millions of vectors while maintaining sub-second query response times.

### Memory and Storage Optimization

Vector storage represents the largest memory requirement in the system. The 384-dimensional embeddings from all-MiniLM-L6-v2 require approximately 1.5KB per chunk, making storage efficiency critical for large document corpora.

The system implements several optimization strategies:
- **Quantization**: Reducing vector precision from float32 to float16 halves storage requirements with minimal accuracy impact
- **Compression**: Qdrant's built-in compression reduces storage overhead for metadata and payload data
- **Tiered Storage**: Frequently accessed vectors remain in memory while older vectors can be stored on disk with acceptable latency impact

### Caching and Performance Tuning

Query performance optimization involves multiple caching layers:

**Embedding Cache**: Frequently queried terms maintain cached embeddings to avoid repeated model inference
**Result Cache**: Common queries cache results for immediate response, with TTL based on data freshness requirements
**Metadata Cache**: Document metadata and relationship graphs cache in memory for fast filtering operations

The system monitors query patterns to optimize cache policies and identify opportunities for precomputation of common result sets.

## Implementation Details and Technical Decisions

### Technology Stack Rationale

**FastAPI** serves as the web framework, chosen for its automatic API documentation generation, type safety, and async support. The async capabilities prove essential for handling concurrent data ingestion and query processing.

**Qdrant** provides vector storage with its combination of performance, ease of deployment, and metadata filtering capabilities. Alternative solutions like Pinecone offer similar functionality but require external hosting.

**Sentence Transformers** enables efficient embedding generation with pre-trained models optimized for semantic similarity tasks. The library's integration with PyTorch allows GPU acceleration when available.

**BeautifulSoup** handles HTML parsing for email content extraction. While alternatives like lxml offer better performance, BeautifulSoup provides more robust handling of malformed HTML commonly found in email messages.

### Error Handling and Reliability

The system implements comprehensive error handling at each pipeline stage:

**Data Ingestion Errors**: API failures, authentication expires, and network timeouts are handled with exponential backoff and circuit breaker patterns. Failed ingestion jobs queue for retry with increasing intervals.

**Processing Errors**: Document parsing failures, embedding generation errors, and vector storage issues are logged with sufficient context for debugging while allowing the pipeline to continue processing other documents.

**Query Errors**: Search failures, timeout conditions, and malformed queries return graceful error responses while maintaining system availability.

### Security and Privacy Considerations

The system handles sensitive information from email and document sources, requiring careful attention to security:

**Authentication Security**: OAuth2 tokens and API keys are stored securely with encryption at rest. The system supports token rotation and revocation.

**Data Privacy**: Document content remains local to the deployment environment. The system doesn't transmit document content to external services beyond the configured data sources.

**Access Control**: While the current implementation assumes single-user access, the architecture supports extension to multi-user scenarios with document-level access controls.

This context engine architecture demonstrates the complexity involved in building production-ready information retrieval systems. The combination of multi-source ingestion, semantic understanding, and contextual awareness creates a foundation for intelligent information access that scales beyond simple keyword matching to true understanding of user intent and information relationships.