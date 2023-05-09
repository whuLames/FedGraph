epoch=1
round=100
echo $epoch
echo $round
for client in 2 4 6 8 10 12 14 16 18 20
do
        process=`expr $client + 1`
        mpirun -n $process python main.py -n $client -r $round -e $epoch
done

