

episodes=2000
for i in 1 2 3
do 
    echo "Seed ${i} - PPO"
    python ppo.py --max_eps=$episodes --seed=$i
    echo "Seed ${i} - PPG"
    python ppg.py --max_eps=$episodes --seed=$i
done

python plot_runs.py