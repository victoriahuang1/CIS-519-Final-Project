if [ $1 -eq 1 ]
then
    echo "Predicting languages..."
    python3.6 language_model.py $2 pred_lm
elif [ $1 -eq 2 ]
then
    echo "Predicting language families..."
    python3.6 language_model.py pred_lm $2 1
else
    echo "Predicting language families..."
    python3.6 language_model.py pred_lm $2 1 1
fi

echo "Computing accuracy..."
python3.6 score.py pred_lm
