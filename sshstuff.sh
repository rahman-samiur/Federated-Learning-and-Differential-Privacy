/HW-2/project cc@129.114.108.135:~/

ssh -i ~/.ssh/hmr-test.pem  cc@129.114.108.135

scp -i ~/.ssh/hmr-test.pem /Users/henriquerio/Documents/IIT/CS595-Project-FedRL/moonlander/experiment.py cc@129.114.108.135:~/moonlander/experiment.py
#copy full moonlander folder to ec2
scp -r -i ~/.ssh/hmr-test.pem /Users/henriquerio/Documents/IIT/CS528-TopSecret-HW/flower-niid-sv-dp cc@129.114.108.135:~/

scp -r -i ~/.ssh/hmr-test.pem /Users/henriquerio/Documents/IIT/CS528-TopSecret-HW/flower-test-sv-dp cc@129.114.108.135:~/

scp -i ~/.ssh/hmr-test.pem /Users/henriquerio/Documents/IIT/CS528-TopSecret-HW/preliminarytesting/dp_sgd_fl_non_iid.py cc@129.114.108.135:~/

scp -i ~/.ssh/hmr-test.pem /Users/henriquerio/Documents/IIT/CS528-TopSecret-HW/preliminarytesting/dp_sgd_fl.py cc@129.114.108.135:~/


#create screen session
screen -S test
#copy npz to sv_results/v1
scp -i ~/.ssh/hmr-test.pem cc@129.114.108.135:~/moonlander/dp_training_data_ep30_sens15.npz  /Users/henriquerio/Documents/IIT/CS595-Project-FedRL/moonlander/sv_results/v1
scp -i ~/.ssh/hmr-test.pem cc@129.114.108.135:~/moonlanderv4/execution_times.txt  /Users/henriquerio/Documents/IIT/CS595-Project-FedRL/sv_results/v4
#continue screen session
screen -r test

#Detach from the screen session: Press Ctrl + a + d.

nohup python3 -u dp_sgd_fl_non_iid.py > output_log_niid.txt 2>&1 &

nohup python3 -u dp_sgd_fl.py > output_log_iid.txt 2>&1 &



#process =37899
# check logs
tail -f output_log_niid.txt
tail -f output_log_iid.txt

# kill all processes
pkill -u cc -9 python3