# Array of languages
languages=("ruby" "javascript" "go" "python" "java" "php")

# Loop through each language
for lang in "${languages[@]}"; do
    python ../evaluator/evaluator.py model_client1/$lang/test_1.gold < model_client1/$lang/test_1.output
done