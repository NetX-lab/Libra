---
dataset_info:
  features:
  - name: repo_name
    dtype: string
  - name: docker_image
    dtype: string
  - name: commit_hash
    dtype: string
  - name: parsed_commit_content
    dtype: string
  - name: execution_result_content
    dtype: string
  - name: modified_files
    sequence: string
  - name: modified_entity_summaries
    list:
    - name: ast_type_str
      dtype: string
    - name: end_lineno
      dtype: int64
    - name: file_name
      dtype: string
    - name: name
      dtype: string
    - name: start_lineno
      dtype: int64
    - name: type
      dtype: string
  - name: relevant_files
    sequence: string
  - name: num_non_test_files
    dtype: int64
  - name: num_non_test_func_methods
    dtype: int64
  - name: num_non_test_lines
    dtype: int64
  - name: prompt
    dtype: string
  - name: problem_statement
    dtype: string
  - name: expected_output_json
    dtype: string
  splits:
  - name: train
    num_bytes: 6079830248
    num_examples: 8101
  download_size: 1662607968
  dataset_size: 6079830248
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
