# PitchPredictor

A python program which uses machine learning to predict which pitch an MLB pitcher will throw next given the game situation (e.g., inning, game score, runners on base. last pitch, etc.)

### Goals

The first goal of this project is to build and train a machine learning model that can predict a pitcher's next pitch. Ideally, this would involve building invididual models for each pitcher (since each pitcher is like a snowflake with their own arsenal of pitches and specific pitches that they are more comfortable with throwing in certain situations).

The second goal of this project is to make our predictions available through a simple (and fast) API-like call. The API should take in raw game data (such as that provided for the training) and return probabilities for each of the pitcher's different pitches. To provide some transparancy, we will also return the accuracy of the model on test data so the user can judge how trustworthy the prediction is.

### The Data

We will use 2011 pitch data from MLB which contains information about every pitch thrown during the year (including the game situation such as the current ball/strike count, score of the game, runners on base as well as more advanced information such as previous pitch locations, speeds and movement).

### Feature Selection and Engineering

Features Selected:

 - Game ID and Pitcher ID: the pitcher ID is included so we can select subsets of the data and build pitcher-specific models. We include the game ID for "groupby" operations.
 
 - Pitch Type: this will be our "outcome" that we are trying to predict. But, we will also use it to tabulate the type of pitch that was last thrown.
Pitch Counts: the pitcher's total pitch count in the game, as well as the pitch count of the particular at-bat.
Game Situation: (1) the inning (and whether it is top or bottom of that inning), (2) balls, strikes & fouls for a particular at-bat, (3) runners on base, (4) visitor and home team runs.
Pitcher-Batter Matchup: (1) handedness of pitcher, (2) handedness of hitter, (3) height of hitter.
Pitch Details: (1) ball, strike or in-play, (2) speed of pitch, (3) break length & angle and (4) zone of the pitch (location inside or outside of the strike zone)
Date of Game: We use the month from the date field to include some seasonality. The reasoning behind this decision is that, as the year progresses, pitchers' arms go through stages where maybe they feel stronger (and are more apt to rely on fastball) or are suffering from "dead arm" (and are more apt to rely on offspeed stuff). Also, weather conditions can affect "feel pitches" such as curveballs, splits and/or change-ups. Since we don't have access to the specific weather conditions of the game, seasonality is the best we can do.
