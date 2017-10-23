basepath=$(cd `dirname $0`; pwd)
cd $basepath
echo Split the dictionary
echo ====================
python ../Split_Dataset/sp.py
echo Split the words with the pronunciations 
echo ====================
python converter.py
echo Save the datasets to $basepath/dataset
echo ====================
python data.py
echo Finished!
