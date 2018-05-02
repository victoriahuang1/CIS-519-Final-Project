if [ $1 -eq 1 ]
then
    echo "Predicting languages..."
    python3.6 language_model.py pred_lm
    echo "Computing accuracy..."
    python3.6 score.py pred_lm
elif [ $1 -eq 2 ]
then
    echo "Predicting language families..."
    python3.6 language_model.py pred_lm 1
    echo "Computing accuracy..."
    python3.6 score.py pred_lm
else
    echo "Predicting language families..."
    python3.6 language_model.py pred_lm 1 1
    echo "Computing accuracy..."
    python3.6 score.py pred_lm
fi
