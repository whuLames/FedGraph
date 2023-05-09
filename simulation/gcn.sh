epoch=10
round=100
datasets=("Cora" "CiteSeer" "CS" "Physics")
echo $epoch
echo $round
for client in 2 4 8 16  
       #	1 2 4 6 8 10 12 14  16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98 100
do
	for ds in "${datasets[@]}" 
	do
        	process=`expr $client + 1`
        	python main.py -n $client -r $round -e $epoch -d $ds
		python main1.py -n $client -r $round -e $epoch -d $ds
		echo $client
	done
done
