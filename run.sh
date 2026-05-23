echo "start"

bash gpt2_1.5B_distill.sh

echo "end train"

sleep 5

bash eval_gpt2_1.5B.sh

echo "end eval"