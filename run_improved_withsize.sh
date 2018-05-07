if [ $1 -eq 1 ]
then
    echo "Predicting languages..."
    python3.6 improved_language_model.py pred_boost $2 0 $3
elif [ $1 -eq 2 ]
then
    echo "Predicting language families..."
    python3.6 improved_language_model.py pred_boost $2 1 $3
fi

echo "Computing accuracy..."
python3.6 score.py pred_boost
