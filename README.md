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
 
 - Pitch Counts: the pitcher's total pitch count in the game, as well as the pitch count of the particular at-bat.

 - Game Situation: (1) the inning (and whether it is top or bottom of that inning), (2) balls, strikes & fouls for a particular at-bat, (3) runners on base, (4) visitor and home team runs.

 - Pitcher-Batter Matchup: (1) handedness of pitcher, (2) handedness of hitter, (3) height of hitter.

 - Pitch Details: (1) ball, strike or in-play, (2) speed of pitch, (3) break length & angle and (4) zone of the pitch (location inside or outside of the strike zone)
 
 - Date of Game: We use the month from the date field to include some seasonality. The reasoning behind this decision is that, as the year progresses, pitchers' arms go through stages where maybe they feel stronger (and are more apt to rely on fastball) or are suffering from "dead arm" (and are more apt to rely on offspeed stuff). Also, weather conditions can affect "feel pitches" such as curveballs, splits and/or change-ups. Since we don't have access to the specific weather conditions of the game, seasonality is the best we can do.
 
Feature Cleaning & Engineering:

 - Drop any rows that are missing values in the "pitch_type" column (i.e., the column we are trying to predict)

 - Extract the month from the game date field (for seasonality reasons)

 - Convert the "on base" fields ("on_1b", "on_2b", "on_3b") to booleans (they are originally populated with player IDs)

 - Construct a boolean feature called "stand_pitch_same_side" that is true if pitcher is throwing from the same side as the hitter is hitting from (and false otherwise)

 - Score differential: difference between the home and visitor runs. We do this in a consistent way such that, if the pitcher's team is winning, the score differential is positive and, if the pitcher's team is losing, the score differential is negative.

 - Construct new features for previous pitch information. To do this, we first make a new ID out of the combination of the game and pitcher IDs. This ensures that we're getting information from the same game. Then, we do groupbys on this new ID and use pandas' "shift" function to grab the previous pitch's speed, type, zone, outcome, and break length/angle.

 - Then, we perform some pitch type cleanup: we condense all of the different types of fastballs (e.g., four-seam, two-seam, sinker, cut fastball, split-finger fastball, etc.) into one type which we call "FB". We also drop any rows that have pitchouts or unknown types of pitches ('PO', 'FO', 'UN', 'XX', 'IN').

 - We map the pitch outcome ('B' for ball, 'S' for strike, 'X' for in-play) to integers for model-building purposes.

### Training Pitcher-Specific Model

Below, we build pitcher-specific models using the function "train_models". This function takes the cleaned dataframe from above as input along with a parameter called "pitch_count_cutoff". Because there are pitchers in this dataset that have a limited amount of pitches thrown during the season, we impose this (somewhat) arbitrarily-chosen cutoff to ensure we are only building meaningful models.

The function loops through the list of pitchers that have a total pitch count above pitch_count_cutoff and:

 - Subsets the dataframe to only include the current pitcher's data

 - Builds a count dictionary of the pitcher's types of pitches

 - Builds a map (and an inverse map) of the pitcher's pitches to integers (and those integers back to the original pitch abbreviations)

 - Splits the data into a dataframe of features (X) and a dataframe of labels (y)

 - Randomly splits X/y into train and test sets using an 80/20 percent training/testing split

 - At this point, we make our choice of algorithm to use to train the model. For the timeframe of this project and the fact that a non-linear model should do a better job in this situation, an XGBoost classifier is an excellent choice. These types of models train fast, tend to have greater accuracy than other non-linear models (including bagging models such as Random Forests) and allow for multi-nomial classification. In particular, we train a multi-nomial version of an XGBoost classifier using "softprob" as the objective function (this will result in each of the possible outcomes being assigned probabilities). 
 
 - We also perform a small grid search over several of the XGBoost hyperparameters using 5-fold cross-validation in order to optimize them. Again, due to time constraints, we cannot do a full hyperparameter optimization but we've chosen to optimize two of the most influential ones ("max_depth" which controls the maximum depth of the tree and, thus, the complexity of the model and "learning_rate" which controls the size of the weights of the features).

 - Once the model has been trained, we perform predictions on the test X and compare to the corresponding test labels y. We also compute what the accuracy would have been had the model just naively chosen the pitcher's most used pitch. These are both stored in lists that we'll use later to assess the overall accuracy of our pitcher-specific models.

 - Finally, we store the trained model along with some metadata (pitcher's ID, the pitch maps and the model accuracy on the test data) in a pickled file. These files will be loaded and used to make predictions in our API.
