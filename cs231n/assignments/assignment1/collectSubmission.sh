#!/bin/bash
#NOTE: DO NOT EDIT THIS FILE-- MAY RESULT IN INCOMPLETE SUBMISSIONS
set -euo pipefail

CODE=(
	"cs231n/classifiers/k_nearest_neighbor.py"
	"cs231n/classifiers/linear_classifier.py"
	"cs231n/classifiers/softmax.py"
	"cs231n/classifiers/fc_net.py"
	"cs231n/optim.py"
	"cs231n/solver.py"
	"cs231n/layers.py"
)

# these notebooks should ideally
# be in order of questions so
# that the generated pdf is
# in order of questions
NOTEBOOKS=(
	"knn.ipynb"
	"softmax.ipynb"
	"two_layer_net.ipynb"
	"features.ipynb"
	"FullyConnectedNets.ipynb"
)

FILES=( "${CODE[@]}" "${NOTEBOOKS[@]}" )

LOCAL_DIR=`pwd`
ASSIGNMENT_NO=1
ZIP_FILENAME="a1_code_submission.zip"
PDF_FILENAME="a1_inline_submission.pdf"

C_R="\e[31m"
C_G="\e[32m"
C_BLD="\e[1m"
C_E="\e[0m"

for FILE in "${FILES[@]}"
do
	if [ ! -f ${FILE} ]; then
		echo -e "${C_R}Required file ${FILE} not found, Exiting.${C_E}"
		exit 0
	fi
done

echo -e "### Zipping file ###"
rm -f ${ZIP_FILENAME}
zip -q "${ZIP_FILENAME}" -r ${NOTEBOOKS[@]} $(find . -name "*.py") "cs231n/saved" -x "makepdf.py"

echo -e "### Creating PDFs ###"
python makepdf.py --notebooks "${NOTEBOOKS[@]}" --pdf_filename "${PDF_FILENAME}"

echo -e "### Done! Please submit ${ZIP_FILENAME} and ${PDF_FILENAME} to Gradescope. ###"
