epoch=5
round=20
echo $epoch
echo $round
for client in 2 6 12 18 24
do
        process=`expr $client + 1`
        mpirun -n $process python main.py -n $client -r $round -e $epoch
done
