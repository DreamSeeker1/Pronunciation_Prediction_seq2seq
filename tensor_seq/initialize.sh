basepath=$(cd `dirname $0`; pwd)
cd $basepath
cd ../Split_Dataset
echo Split the dictionary
echo ====================
python sp.py
echo Split the words with the pronunciations 
echo ====================
cd ../tensor_seq
python converter.py
echo Save the datasets to $basepath/dataset
echo ====================
python data.py
echo Finished!
