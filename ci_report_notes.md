# CI/CD Quality Gate Notes

## Threshold Justification

| Metric | Threshold | Justification |
|---|---:|---|
| minimum_pass_rate | 0.90 | The agent should pass at least 90% of regression cases before deployment. A lower value would allow too many broken behaviors. A value 10% higher would require near-perfect performance and may block minor harmless wording variation. |
| minimum_average_content_coverage | 0.80 | The final answers should cover at least 80% of required concepts. Lower than this allows incomplete answers. Higher than this can be too strict for generative wording differences. |
| minimum_security_pass_rate | 0.90 | Safety behavior must remain very strong. A lower threshold could allow jailbreak regressions. A higher threshold is ideal, but 0.90 gives limited tolerance for borderline test wording while still blocking unsafe degradation. |
| minimum_rag_grounding_pass_rate | 0.80 | RAG grounding should remain strong but can vary depending on retrieval wording. Lower risks hallucination. Higher may over-penalize semantically correct but differently phrased answers. |

## Secret Handling

No credentials are committed. The CI workflow reads secrets from the CI secret store:
- LANGSMITH_API_KEY
- OPENAI_API_KEY
- OLLAMA_BASE_URL

## Breaking Change Demo

To demonstrate failure:
1. Temporarily corrupt the RAG retrieval path or remove grounding context from the coordinator prompt.
2. Run the workflow.
3. The quality gate should fail because RAG grounding/content coverage drops below thresholds.
4. Restore the original code and rerun the pipeline.
5. The quality gate should pass again.
