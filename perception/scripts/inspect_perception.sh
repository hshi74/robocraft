tool_type="gripper_sym_rod_robot_v1"
perception_dir="./dump/perception/gripper_sym_rod_robot_v1_04-Jan-2023-14:36:42.790368"

mkdir -p ./dump/perception/inspect/$tool_type

# echo $perception_dir
for sub_dir in $(find $perception_dir -maxdepth 1 -type d); do
    file="$sub_dir/repr.mp4"
    if test -f "$file"; then        
        echo $sub_dir
        vid_idx=$(basename -- "$sub_dir")
        cp $file ./dump/perception/inspect/$tool_type/$vid_idx.mp4
    fi
done

# touch ./dump/perception/inspect/inspect.txt