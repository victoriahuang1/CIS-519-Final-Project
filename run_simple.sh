if [ $1 -eq 1 ]
then
    echo "Predicting languages..."
    python3.6 simple_baseline.py pred
    echo "Computing accuracy..."
    python3.6 score.py pred
elif [ $1 -eq 2 ]
then
    echo "Predicting language families..."
    python3.6 simple_baseline.py pred 1
    echo "Computing accuracy..."
    python3.6 score.py pred
else
    echo "Predicting language families..."
    python3.6 simple_baseline.py pred 1 1
    echo "Computing accuracy..."
    python3.6 score.py pred
fi
