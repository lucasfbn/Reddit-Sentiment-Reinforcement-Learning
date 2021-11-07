## Find optimal parameters for preprocessing, [#106](https://github.com/lucasfbn/Trendstuff/issues/106)

- Objectives

  - Find
    - [ ] ticker_min_len
    - [ ] sequence_len
    - [ ] min_sequence_len
  - Abbreviated from above-mentioned
    - [ ] price_data_start_offset

- Steps

  - [x] Make dashboard

    - [x] Plot timeseries

    - [x] Plot time lagged cross correlation

    - [x] Plot max tlcc of common combinations

    - [x] Plot bulk data statistics

    - [x] Make sure interpretation is correct

      *Yes. Can be tracked by shifting a dataframe:*
      
      ```python
      df = pd.Series(range(10))
      df = df.shift(-2)
      ```
      
      *The resulting dataframe will be shifted "backwards". E.g. the cross correlation will also be calculated with the backwards shifted y timeseries.*





