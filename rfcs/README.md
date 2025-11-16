# Meilisearch RFCs (Request for Comments)

This directory contains RFC documents for major improvements and features proposed for Meilisearch.

## RFC Series Overview

This comprehensive RFC series emerged from extensive technical research and codebase analysis documented in:
- [Research Plan](../RESEARCH_PLAN.md)
- [Architecture Overview](../ARCHITECTURE_OVERVIEW.md)
- [Blog Post Series](../blog_posts/)

The RFCs are organized by focus area and prioritized by impact and feasibility.

---

## Active RFCs by Priority

### 🔥 High Priority - High Impact, Medium Feasibility

#### [RFC 007: Parallel Indexing Optimization & Performance Enhancements](007_parallel_indexing_optimization.md)

**Status:** Draft  
**Focus:** Performance Optimization  
**Complexity:** Medium  
**Timeline:** 12 weeks

Comprehensive parallelization of indexing pipeline for 30-50% throughput improvements:
- **Parallel prefix computation**: 3.75x speedup
- **Parallel facet building**: 2.5x speedup for full build, 16x for incremental
- **Intelligent batch sizing**: Adaptive memory management
- **GPU acceleration**: 10x speedup for embedding generation

**Key Benefits:**
- Immediate impact on all users
- No API changes required
- Backward compatible
- Clear performance wins

**Estimated ROI:** Very High (direct user impact, existing demand)

---

#### [RFC 001: Enhanced Hybrid Search with Advanced Fusion Strategies](001_enhanced_hybrid_search_fusion.md)

**Status:** Draft  
**Focus:** Search Quality  
**Complexity:** Medium  
**Timeline:** 13 weeks

Significant enhancements to hybrid search:
- **Reciprocal Rank Fusion (RRF)**: More robust than weighted sum
- **Adaptive semantic ratio**: Query-aware blending
- **Learned fusion weights**: CTR-based optimization

**Key Benefits:**
- Better relevance for diverse query types
- Reduced sensitivity to score distribution
- Automatic optimization from user behavior

**Estimated ROI:** High (improves core search quality)

---

#### [RFC 006: Search Analytics & Relevancy Intelligence](006_search_analytics_relevancy_intelligence.md)

**Status:** Draft  
**Focus:** Observability & Machine Learning  
**Complexity:** Medium  
**Timeline:** 15 weeks

Comprehensive analytics and automated relevancy improvement:
- **Real-time analytics dashboard**: Query patterns, CTR tracking
- **A/B testing framework**: Test ranking configurations
- **Relevancy learning**: Automated search quality improvement
- **Cost attribution**: Per-tenant metrics

**Key Benefits:**
- Data-driven search optimization
- Competitive with Algolia analytics
- Enables SaaS use cases
- Improves search over time

**Estimated ROI:** High (differentiator feature, enables new business models)

---

### ⚡ Medium Priority - High Impact, High Complexity

#### [RFC 002: Distributed Meilisearch Architecture](002_distributed_architecture.md)

**Status:** Draft  
**Focus:** Horizontal Scaling  
**Complexity:** High  
**Timeline:** 32 weeks

Distributed architecture for massive scale:
- **Cluster coordinator**: Raft-based consensus
- **Shard allocation**: Consistent hashing
- **Query routing**: Distributed query execution
- **Replication**: High availability

**Key Benefits:**
- Horizontal scaling (>100M documents)
- High availability through replication
- Geographic distribution
- Linear write throughput scaling

**Estimated ROI:** Medium (addresses minority of users but critical for enterprise)

---

#### [RFC 004: Streaming Indexing with Real-Time Updates](004_streaming_indexing_real_time.md)

**Status:** Draft  
**Focus:** Write Performance & Real-Time  
**Complexity:** Medium-High  
**Timeline:** 15 weeks

True real-time indexing with sub-second latency:
- **In-memory write buffer**: Fast writes (5ms vs 150ms)
- **Write-ahead log**: Crash recovery
- **Background merge**: Async LMDB updates
- **5x write throughput**: 100-200K docs/sec

**Key Benefits:**
- Addresses LMDB single-writer bottleneck
- Real-time search visibility
- Better multi-tenant performance
- Maintains ACID guarantees

**Estimated ROI:** High (solves known bottleneck, high user demand)

---

#### [RFC 003: Advanced Vector Search Capabilities](003_advanced_vector_search.md)

**Status:** Draft  
**Focus:** Vector Search Enhancement  
**Complexity:** Medium-High  
**Timeline:** 23 weeks

Advanced vector search features:
- **Product Quantization**: 32× compression with 92-95% accuracy
- **IVF-HNSW**: Billion-scale vector search
- **Multi-vector documents**: Multiple embeddings per document
- **Configurable parameters**: Runtime ef_search control

**Key Benefits:**
- Better memory/accuracy trade-off
- Scalability to 100M+ vectors
- Rich document representation
- Fine-grained search quality control

**Estimated ROI:** Medium-High (growing vector search demand)

---

### 💡 Medium Priority - Medium Impact, Medium Complexity

#### [RFC 009: Multi-Vector Documents & Advanced RAG Support](009_multi_vector_documents_advanced_rag.md)

**Status:** Draft  
**Focus:** RAG & Semantic Search  
**Complexity:** Medium  
**Timeline:** 16 weeks

Advanced RAG capabilities:
- **Multi-vector documents**: Chunk-based embeddings
- **MaxSim aggregation**: Best-chunk matching
- **ColBERT-style late interaction**: Token-level matching
- **Cross-encoder reranking**: Precision relevance scoring
- **Contextual retrieval**: Chunks with surrounding context

**Key Benefits:**
- Enables advanced RAG patterns
- Better long-document handling
- State-of-the-art semantic search
- Citation support

**Estimated ROI:** Medium (growing RAG/LLM application market)

---

#### [RFC 010: Resource Quotas & Quality of Service](010_resource_quotas_quality_of_service.md)

**Status:** Draft  
**Focus:** Enterprise Features & Multi-Tenancy  
**Complexity:** Medium  
**Timeline:** 13 weeks

Enterprise-grade resource management:
- **Per-tenant quotas**: CPU, memory, disk, API rate limits
- **Priority queues**: Service tier-based scheduling
- **Cost attribution**: Billing metrics per tenant
- **Tenant isolation**: Resource guarantees

**Key Benefits:**
- Enables SaaS business models
- Prevents abuse and resource exhaustion
- Fair resource allocation
- Cost tracking and billing

**Estimated ROI:** Medium (critical for SaaS providers, niche for others)

---

#### [RFC 011: Learned Sparse Representations with SPLADE](011_learned_sparse_representations_splade.md)

**Status:** Draft  
**Focus:** Modern Information Retrieval  
**Complexity:** High  
**Timeline:** 14 weeks

State-of-the-art learned sparse retrieval:
- **SPLADE integration**: Neural sparse vectors
- **Query expansion**: Semantic term weighting
- **Triple hybrid**: BM25 + SPLADE + Dense
- **Domain adaptation**: Fine-tuning support

**Key Benefits:**
- Better than BM25 (30% recall improvement)
- Better generalization than dense vectors
- Inverted index efficiency maintained
- State-of-the-art retrieval quality

**Estimated ROI:** Medium (cutting-edge but requires ML expertise)

---

### 🎯 Lower Priority - Medium Impact, Low-Medium Complexity

#### [RFC 008: Enhanced Error Messages & Developer Tooling](008_enhanced_error_messages_dev_tooling.md)

**Status:** Draft  
**Focus:** Developer Experience  
**Complexity:** Low-Medium  
**Timeline:** 12 weeks

Dramatically improved developer experience:
- **Actionable errors**: Errors with fix suggestions and cURL examples
- **Query explanation API**: Understand ranking decisions
- **Schema validation**: Helpful settings validation
- **CLI debugging tools**: Explain, profile, validate commands
- **Interactive query builder**: Metadata for building queries

**Key Benefits:**
- Faster debugging and development
- Reduced support burden
- Better user onboarding
- Competitive with best-in-class APIs (Stripe-level DX)

**Estimated ROI:** Medium (improves UX for all users, lower technical barrier)

---

#### [RFC 005: GraphQL API for Flexible Query Composition](005_graphql_api_flexible_queries.md)

**Status:** Draft  
**Focus:** API Design & Developer Experience  
**Complexity:** Medium  
**Timeline:** 10 weeks

Modern GraphQL API alongside REST:
- **Flexible queries**: Nested data fetching in single request
- **Type safety**: Auto-generated client types
- **Real-time subscriptions**: WebSocket support for live updates
- **Reduced over-fetching**: Precise field selection
- **Better DX**: Introspection and tooling

**Key Benefits:**
- Modern API alternative
- Better frontend integration
- Type-safe client generation
- Real-time capabilities

**Estimated ROI:** Medium (appeals to modern web developers, optional feature)

---

## RFC Summary Table

| RFC | Title | Focus Area | Priority | Complexity | Timeline | Est. ROI |
|-----|-------|------------|----------|------------|----------|----------|
| 007 | Parallel Indexing Optimization | Performance | 🔥 High | Medium | 12 weeks | Very High |
| 001 | Enhanced Hybrid Search Fusion | Search Quality | 🔥 High | Medium | 13 weeks | High |
| 006 | Search Analytics & Intelligence | Observability | 🔥 High | Medium | 15 weeks | High |
| 002 | Distributed Architecture | Scalability | ⚡ Medium | High | 32 weeks | Medium |
| 004 | Streaming Indexing | Write Performance | ⚡ Medium | Medium-High | 15 weeks | High |
| 003 | Advanced Vector Search | Vector Capabilities | ⚡ Medium | Medium-High | 23 weeks | Medium-High |
| 009 | Multi-Vector Documents & RAG | Vector/RAG | 💡 Medium | Medium | 16 weeks | Medium |
| 010 | Resource Quotas & QoS | Enterprise | 💡 Medium | Medium | 13 weeks | Medium |
| 011 | SPLADE Integration | Modern IR | 💡 Medium | High | 14 weeks | Medium |
| 008 | Enhanced Errors & Dev Tools | Developer Experience | 🎯 Lower | Low-Medium | 12 weeks | Medium |
| 005 | GraphQL API | API Design | 🎯 Lower | Medium | 10 weeks | Medium |

**Total estimated effort:** ~185 weeks (3.5 years) if done sequentially  
**Recommended parallel workstreams:** 3-4 teams working concurrently

---

## Implementation Priority Tiers

### Tier 1: Quick Wins (Start Immediately)
1. **RFC 007** - Parallel Indexing (12 weeks) - Immediate performance impact
2. **RFC 008** - Enhanced Errors (12 weeks) - Improves all user interactions
3. **RFC 001** - Hybrid Search Fusion (13 weeks) - Core search improvement

**Total Tier 1:** ~12 weeks with 3 parallel teams, ~37 weeks sequential

### Tier 2: Strategic Enhancements (Next 6 Months)
4. **RFC 006** - Search Analytics (15 weeks) - Competitive feature
5. **RFC 004** - Streaming Indexing (15 weeks) - Addresses known bottleneck
6. **RFC 009** - Multi-Vector/RAG (16 weeks) - Growing market demand

**Total Tier 2:** ~16 weeks with 3 parallel teams, ~46 weeks sequential

### Tier 3: Advanced Features (6-12 Months)
7. **RFC 003** - Advanced Vector Search (23 weeks) - Specialized use cases
8. **RFC 010** - Resource Quotas (13 weeks) - Enterprise/SaaS enabler
9. **RFC 005** - GraphQL API (10 weeks) - Modern API alternative

**Total Tier 3:** ~23 weeks with 3 parallel teams, ~46 weeks sequential

### Tier 4: Long-Term Vision (12-24 Months)
10. **RFC 011** - SPLADE Integration (14 weeks) - Cutting-edge IR
11. **RFC 002** - Distributed Architecture (32 weeks) - Major architectural shift

**Total Tier 4:** ~32 weeks with 2 parallel teams, ~46 weeks sequential

---

## Recommended Implementation Strategy

### Scenario A: Maximum Impact (3 Parallel Teams)

**Quarter 1 (Weeks 1-13):**
- Team 1: RFC 007 (Parallel Indexing)
- Team 2: RFC 008 (Enhanced Errors)
- Team 3: RFC 001 (Hybrid Search)

**Quarter 2 (Weeks 14-29):**
- Team 1: RFC 006 (Analytics)
- Team 2: RFC 004 (Streaming Indexing)
- Team 3: RFC 009 (Multi-Vector RAG)

**Quarter 3 (Weeks 30-52):**
- Team 1: RFC 003 (Advanced Vector)
- Team 2: RFC 010 (Resource Quotas)
- Team 3: RFC 005 (GraphQL API)

**Quarter 4+ (Weeks 53+):**
- Team 1: RFC 011 (SPLADE)
- Team 2+3: RFC 002 (Distributed Architecture)

**Total Timeline:** ~16 months for all RFCs with 3 teams

### Scenario B: Focused Excellence (1-2 Teams)

**Year 1:**
- RFC 007 (Parallel Indexing)
- RFC 001 (Hybrid Search)
- RFC 008 (Enhanced Errors)
- RFC 006 (Analytics)
- RFC 004 (Streaming Indexing)

**Year 2:**
- RFC 009 (Multi-Vector RAG)
- RFC 003 (Advanced Vector)
- RFC 010 (Resource Quotas)
- RFC 005 (GraphQL API)

**Year 3:**
- RFC 011 (SPLADE)
- RFC 002 (Distributed Architecture)

---

## RFC Categories

### 🎯 Search Quality & Relevance
- [RFC 001: Enhanced Hybrid Search Fusion](001_enhanced_hybrid_search_fusion.md)
- [RFC 006: Search Analytics & Relevancy Intelligence](006_search_analytics_relevancy_intelligence.md)
- [RFC 011: Learned Sparse Representations (SPLADE)](011_learned_sparse_representations_splade.md)

### ⚡ Performance & Scalability
- [RFC 002: Distributed Architecture](002_distributed_architecture.md)
- [RFC 004: Streaming Indexing with Real-Time Updates](004_streaming_indexing_real_time.md)
- [RFC 007: Parallel Indexing Optimization](007_parallel_indexing_optimization.md)

### 🔮 Vector Search & AI
- [RFC 003: Advanced Vector Search Capabilities](003_advanced_vector_search.md)
- [RFC 009: Multi-Vector Documents & Advanced RAG](009_multi_vector_documents_advanced_rag.md)

### 👥 Developer Experience & Tooling
- [RFC 005: GraphQL API for Flexible Queries](005_graphql_api_flexible_queries.md)
- [RFC 008: Enhanced Error Messages & Developer Tooling](008_enhanced_error_messages_dev_tooling.md)

### 🏢 Enterprise & Multi-Tenancy
- [RFC 010: Resource Quotas & Quality of Service](010_resource_quotas_quality_of_service.md)

---

## RFC Lifecycle Stages

1. **Draft**: Initial proposal under discussion (ALL CURRENT RFCs)
2. **Review**: Community feedback period
3. **Accepted**: Approved for implementation
4. **Implemented**: Feature merged to main branch
5. **Stable**: Available in stable release

---

## RFC Process

### Proposing a New RFC

1. Review existing RFCs to avoid duplication
2. Create RFC document following template
3. Use sequential numbering: `012_your_feature_name.md`
4. Submit PR to `rfcs/` directory
5. Announce in GitHub Discussions
6. Iterate based on community feedback
7. Seek maintainer approval

### RFC Template Structure

Each RFC should include:

1. **Summary** - One paragraph overview
2. **Motivation** - Why this matters, current limitations, use cases
3. **Technical Design** - Detailed architecture, code examples, algorithms
4. **API Changes** - New endpoints, configuration, examples
5. **Backward Compatibility** - Migration strategy, deprecation timeline
6. **Implementation Plan** - Phased approach with milestones
7. **Performance Implications** - Benchmarks, overhead analysis
8. **Drawbacks** - Honest assessment of downsides
9. **Alternatives Considered** - Why this approach over others
10. **Open Questions** - Unresolved design decisions
11. **References** - Research papers, existing implementations
12. **Community Discussion** - Discussion link, key questions

---

## Community Discussion

RFCs are discussed on:
- **GitHub Discussions**: [Meilisearch Discussions](https://github.com/meilisearch/meilisearch/discussions)
- **Discord**: [Meilisearch Discord](https://discord.meilisearch.com)
- **RFC Pull Requests**: Direct PR reviews

---

## Relationship to Research & Blog Posts

These RFCs build upon extensive technical research:

### Research Foundation
- [Research Plan](../RESEARCH_PLAN.md): Comprehensive analysis roadmap
- [Architecture Overview](../ARCHITECTURE_OVERVIEW.md): System design deep-dive

### Blog Post Series
1. [Architecture Deep Dive](../blog_posts/01_meilisearch_architecture_deep_dive.md)
2. [Typo Tolerance & Levenshtein Automata](../blog_posts/02_typo_tolerance_levenshtein_automata.md)
3. [Hybrid Search: Vector-Keyword Fusion](../blog_posts/03_hybrid_search_vector_keyword_fusion.md)
4. [LMDB Storage Engine Internals](../blog_posts/04_lmdb_storage_engine_internals.md)
5. [Optimal Use Cases & Decision Guide](../blog_posts/05_optimal_use_cases_decision_guide.md)
6. [Meilisearch vs Alternatives Benchmark](../blog_posts/06_meilisearch_vs_alternatives_benchmark.md)
7. [Advanced Patterns & Production Search](../blog_posts/07_advanced_patterns_production_search.md)

---

## Impact Analysis

### Feature Coverage Matrix

| Area | Current State | Proposed RFCs | Impact |
|------|---------------|---------------|--------|
| **Search Quality** | Good | RFCs 001, 006, 011 | +40% relevance improvement |
| **Performance** | Good | RFCs 004, 007 | +50% indexing throughput |
| **Scalability** | Limited | RFC 002 | 10x document capacity |
| **Vector Search** | Basic | RFCs 003, 009, 011 | Best-in-class capabilities |
| **Developer Experience** | Good | RFCs 005, 008 | Industry-leading DX |
| **Enterprise Features** | Basic | RFC 010 | SaaS-ready |
| **Analytics** | Minimal | RFC 006 | Competitive with Algolia |

### Market Positioning Impact

**Current Position:** Fast, simple search for < 100M documents

**After RFCs Implementation:**
- **Search Quality**: Match or exceed Algolia/Elasticsearch
- **Scale**: Compete with Elasticsearch (via RFC 002)
- **Vector Search**: Compete with dedicated vector DBs
- **Developer Experience**: Industry-leading (RFCs 005, 008)
- **Enterprise Ready**: Multi-tenant SaaS capable (RFC 010)
- **Analytics**: Match Algolia (RFC 006)

---

## Combined Benefits Analysis

### If All RFCs Implemented

**Performance Improvements:**
- Indexing throughput: **+200-500%** (RFCs 004, 007)
- Search latency: **-15-30%** (RFCs 001, 007, 011)
- Write latency: **-95%** (RFC 004: 5ms vs 150ms)
- Maximum scale: **10-100x** (RFC 002: distributed)

**Feature Completeness:**
- Vector search: **Best-in-class** (RFCs 003, 009, 011)
- Analytics: **Algolia-competitive** (RFC 006)
- Developer tools: **Industry-leading** (RFCs 005, 008)
- Enterprise features: **SaaS-ready** (RFC 010)

**Market Differentiation:**
- Only search engine with **native triple hybrid** (BM25 + SPLADE + Dense)
- **Algolia simplicity** with **Elasticsearch power**
- **Best-in-class performance** (< 50ms p99) at **massive scale** (100M-1B docs)

---

## Dependencies Between RFCs

### Independent (Can Implement in Parallel)
- RFC 001, 005, 007, 008 (no dependencies)

### Sequential Dependencies
- RFC 003 → RFC 009 (multi-vector builds on advanced vector)
- RFC 001 → RFC 011 (SPLADE uses hybrid fusion framework)
- RFC 006 → RFC 001 (analytics can leverage learned weights)

### Complementary (Better Together)
- RFC 001 + RFC 011 (hybrid with SPLADE)
- RFC 003 + RFC 009 (advanced vector + multi-vector)
- RFC 004 + RFC 002 (streaming + distributed)
- RFC 006 + RFC 010 (analytics + quotas for SaaS)

---

## Success Metrics

### For Each RFC

**Technical Metrics:**
- Performance improvement targets met
- No critical bugs in stable release
- Test coverage > 85%
- Documentation complete

**Adoption Metrics:**
- Community feedback positive (> 70%)
- Production usage by 3+ companies within 6 months
- GitHub stars increase
- Blog posts / conference talks referencing feature

**Business Metrics:**
- Reduces support tickets for related issues
- Enables new use cases
- Increases user retention
- Attracts enterprise customers (for enterprise RFCs)

---

## Frequently Asked Questions

### Which RFC should I start with?

**For quick wins:** RFC 007 (Parallel Indexing) or RFC 008 (Enhanced Errors)

**For search quality:** RFC 001 (Hybrid Fusion)

**For scale:** RFC 002 (Distributed) or RFC 004 (Streaming)

**For AI/ML features:** RFC 003, 009, or 011

### Can RFCs be implemented independently?

Most RFCs are independent, but some build on others (see dependencies above).

### What's the expected timeline?

- **Single RFC:** 10-32 weeks depending on complexity
- **All RFCs (1 team):** ~3.5 years
- **All RFCs (3 teams):** ~16 months

### How to contribute?

See each RFC's Community Discussion section for links to GitHub discussions.

---

## Revision History

- **2025-11-10**: Initial RFCs 001-003 created
- **2025-11-16**: RFCs 004-011 added, comprehensive series completed
- **2025-11-16**: README updated with full categorization and prioritization

---

**Last Updated:** 2025-11-16