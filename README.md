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

## Implement working RL agent

- Objectives

  - [ ] Realize working RL agent

- Steps

- Ideas

  1. Check if 1D-CNN is applied correctly
     - Time series wrong direction?

  2. Use Random Search combined with Supervised Learning
  3. Use different implementation? (for instance: [Stable Baselines](https://github.com/Stable-Baselines-Team/stable-baselines/tree/master/stable_baselines/ppo2))

- Paper read

  - [Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996)
    - PPO best overall result
    - Reward function is nothing fancy, they are using the change in the balance as reward
    - They trade several different stocks by combining them in a single matrices (but the number of stocks are fixed with this approach)
    - Implementation [available](https://github.com/AI4Finance-Foundation/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020)

- Paper to be read

  - [Application of Deep Reinforcement Learning in Stock Trading Strategies and Stock Forecasting](https://core.ac.uk/download/pdf/322327703.pdf)





