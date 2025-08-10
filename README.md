# NFL play-by-play EDA

> Exploratory data analysis of NFL play-by-play data using nfl-data-py

## Contents

- [NFL play-by-play EDA](#nfl-play-by-play-eda)
  - [Contents](#contents)
  - [Playcalling](#playcalling)

## Playcalling

The first goal is to develop a playcalling engine based on real NFL playcalling data.  Initially the playcalling engine might be based purely on game context, such as
- Score (point differential)
- Time
- Quarter
- Yard line
- Down & distance
- Timeouts remaining

In the future, we might extend this playcalling engine to accept additional team context, such as
- Offensive playcalling style
- Passing game overall
  - Pass blocking
  - QB overall
  - WR overall
- Rushing game overall
  - Run blocking
  - RB overall
