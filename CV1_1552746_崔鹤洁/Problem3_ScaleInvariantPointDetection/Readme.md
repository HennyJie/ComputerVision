### Dependency

- python 3.5
- opencv 3.x
- scipy 

###Usage

```shell
#Directly run the code with default parameters:
python main.py

#Change the parameters:
python main.py --help

optional arguments:
  -h, --help            show this help message and exit
  -p, --picture_path    Directory where to get the source picture.
  -i, --sigma_init      The initial value of sigma.
  -f, --sigma_final     The largest scale to process.
  -s, --s               The number of scales for calculating the          						  extreme value.
  -t, --threshold       Laplacian threshold.

```



### Output

![mage-20180407223948](/var/folders/6h/c166vcld5hq6lf3vkh1v_x8m0000gn/T/abnerworks.Typora/image-201804072239485.png)



