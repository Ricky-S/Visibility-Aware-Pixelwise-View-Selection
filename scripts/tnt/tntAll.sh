# Intermediate

# Family
# Francis
# Horse
# Lighthouse
# M60
# Panther
# Playground
# Train247

# Advanced

# Auditorium			
# Ballroom			
# Courtroom			
# Museum			
# Palace			
# Temple

version="0818"

# Sample images
nohup bash scripts/tnt/tntSingle.sh "tnt_In_Horse-$version"         "intermediate/Horse"        > results/tnt/"tnt_Int_Horse-$version".log 2>&1 &
wait
exit 0

# 150 images
nohup bash scripts/tnt/tntSingle.sh "tnt_In_Family-$version"        "intermediate/Family"       > results/tnt/"tnt_Int_Family-$version".log 2>&1 &
wait
nohup bash scripts/tnt/tntSingle.sh "tnt_In_Horse-$version"         "intermediate/Horse"        > results/tnt/"tnt_Int_Horse-$version".log 2>&1 &
wait
# 300 images
wait
nohup bash scripts/tnt/tntSingle.sh "tnt_In_Francis-$version"       "intermediate/Francis"      > results/tnt/"tnt_Int_Francis-$version".log 2>&1 &
wait
nohup bash scripts/tnt/tntSingle.sh "tnt_In_Lighthouse-$version"    "intermediate/Lighthouse"   > results/tnt/"tnt_Int_Lighthouse-$version".log 2>&1 &
wait
nohup bash scripts/tnt/tntSingle.sh "tnt_In_M60-$version"           "intermediate/M60"          > results/tnt/"tnt_Int_M60-$version".log 2>&1 &
wait
nohup bash scripts/tnt/tntSingle.sh "tnt_In_Panther-$version"       "intermediate/Panther"      > results/tnt/"tnt_Int_Panther-$version".log 2>&1 &
wait
nohup bash scripts/tnt/tntSingle.sh "tnt_In_Playground-$version"    "intermediate/Playground"   > results/tnt/"tnt_Int_Playground-$version".log 2>&1 &
wait
nohup bash scripts/tnt/tntSingle.sh "tnt_In_Train-$version"         "intermediate/Train"        > results/tnt/"tnt_Int_Train-$version".log 2>&1 &
wait

# Advanced images
nohup bash scripts/tnt/tntSingle.sh "tnt_Ad_Auditorium-$version"    "advanced/Auditorium"       > results/tnt/"tnt_Ad_Auditorium-$version".log 2>&1 &
wait
nohup bash scripts/tnt/tntSingle.sh "tnt_Ad_Ballroom-$version"      "advanced/Ballroom"         > results/tnt/"tnt_Ad_Ballroom-$version".log 2>&1 &
wait
nohup bash scripts/tnt/tntSingle.sh "tnt_Ad_Courtroom-$version"     "advanced/Courtroom"        > results/tnt/"tnt_Ad_Courtroom-$version".log 2>&1 &
wait
nohup bash scripts/tnt/tntSingle.sh "tnt_Ad_Museum-$version"        "advanced/Museum"           > results/tnt/"tnt_Ad_Museum-$version".log 2>&1 &
wait
nohup bash scripts/tnt/tntSingle.sh "tnt_Ad_Temple-$version"        "advanced/Temple"           > results/tnt/"tnt_Ad_Temple-$version".log 2>&1 &
wait
# 500 images
nohup bash scripts/tnt/tntSingle.sh "tnt_Ad_Palace-$version"        "advanced/Palace"           > results/tnt/"tnt_Ad_Palace-$version".log 2>&1 &
wait
exit 0




