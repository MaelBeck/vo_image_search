import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import sys
from timeit import default_timer as timer

'''
    Usage:
    ./query_search.py -d "database_name" -q "query_imagename" -t "index_type" -r "relevant_images_number"
    
    Example:
    python query_search.py -d COREL -q corel_0000000303_512 -t LINEAR
'''


######## Program parameters
import argparse
parser = argparse.ArgumentParser()

## Database name
parser.add_argument("-d", "--database", dest="db_name",
                    help="input image database", metavar="STRING")


## Query image name
parser.add_argument("-q", "--query", dest="query_name",
                    help="query image name", metavar="STRING")

## Database Index Type
parser.add_argument("-t", "--indextype", dest="indextype",
                    help="index type", metavar="STRING")

## Number of relevant images in the database, considering the query
parser.add_argument("-r", "--relevant", dest="relevant",  type=int,
                    help="relevant image number", metavar="INTEGER", default=4)

args = parser.parse_args()


## Set paths
img_dir="/share/esir3/VO/Images/" + args.db_name + "/"
if (args.db_name == "COREL" || args.db_name == "NISTER" ):
    img_dir="/share/esir3/VO/Images/" + args.db_name + "_queries/"
db_dir="./databases/"+ args.db_name
resfilename = "./results/" + args.query_name + "-" + args.indextype



## Load query image
query_filename=img_dir + args.query_name + ".jpg"
if not os.path.isfile(query_filename):
    print "Path to the query "+query_filename+" is not found -- EXIT\n"
    sys.exit(1)

queryImage = cv2.imread(query_filename)

plt.figure(0), plt.title("Image requete")
plt.imshow(cv2.cvtColor(queryImage, cv2.COLOR_BGR2RGB))
plt.show()

# Load descriptors
dataBaseDesc = np.load(db_dir+"_dbDesc.npy")
imagesIndexNames = np.load(db_dir+"_imageIndex")

# Load index
index_params = dict(algorithm=254,filename=db_dir+"_"+args.indextype+"Index")

fl = cv2.flann_Index(np.asarray(dbDesc,np.float32),index_params)



#########################
#### Display the top images
#########################
top=10
plt.figure(1), plt.title(args.indextype )
for i in range(top):
    img = cv2.imread(filtered_scores[i][1])
    score = filtered_scores[i][0]
    plt.subplot(2,5,i+1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('rank '+str(i+1)), plt.xticks([]), plt.yticks([]),plt.xlabel(str(score))

plt.savefig(resfilename + "_top" + str(top) +".png")
plt.show()



########################
### Evaluation : TO COMPLETE
########################


def getImageId(imname):
    if (args.db_name == "COREL"):
        Id = imname.split('_')[1]
    elif (args.db_name == "NISTER"):
        Id = imname.split('-')[1]
    elif (args.db_name == "Copydays"):
        Id = imname.split('_')[-2]
    else:
        Id = imname.split('.')[-1]
    
    return Id


queryId=getImageId(args.query_name)
print "Identifiant de la requete : ", queryId

rpFile = open(resfilename + "_rp.dat", 'w')
precision = np.zeros(len(filtered_scores), dtype=float)
recall = np.zeros(len(filtered_scores), dtype=float)

nbRelevantImage = args.relevant
#[...]

for tuple in filtered_scores:
    #[...]
    #if(getImageId(tuple[1]) == queryId):
            #[...]
    #[...]
    #precision[...] =
    #recall[...] =
    #rpFile.write(str(precision[]) + '\t' + str(recall[]) +  '\n')


# Plot Precision-Recall curve
plt.clf()
plt.plot(recall, precision, lw=2, color='navy',
         label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.05])
plt.title('Precision-Recall for '+args.query_name)
plt.legend(loc="upper right")
plt.savefig(resfilename + "_rp.png")
#plt.savefig(resfilename + "_rp.pdf", format='pdf')
plt.show()




