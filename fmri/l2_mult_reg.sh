

task=(1 2)

#python /om/user/ysa/l2_mult_regres.py -o /om/user/ysa/bild/data/second_level_TD_motion_removed_All_43_z2.57/output_dir -t 1 -d /om/user/ysa/bild/data -l1 /om/user/ysa/bild/data/first_level/output_dir -w /om/user/ysa/bild/data/second_level_TD_motion_removed_All_43_z2.57/working_dir_1 -p 'SLURM' --plugin_args "dict(sbatch_args='-N1 -c2 --mem=10G')" --sleep 2 -f ols

#python /om/user/ysa/l2_mult_regres.py -o /om/user/ysa/bild/data/second_level_TD_task2_syntaxOutlierRemoved_z1.96/output_dir -t 2 -d /om/user/ysa/bild/data -l1 /om/user/ysa/bild/data/first_level/output_dir -w /om/user/ysa/bild/data/second_level_TD_task2_syntaxOutlierRemoved_z1.96/working_dir_2 -p 'SLURM' --plugin_args "dict(sbatch_args='-N1 -c2 --mem=10G')" --sleep 2 -f ols



#used for kids and adults task1
python /om/user/ysa/l2_mult_regres.py -o /om/user/ysa/bild/data/second_level_TD_KidsAdults_task1/output_dir -t 1 -d /om/user/ysa/bild/data -l1 /om/user/ysa/bild/data/first_level/output_dir -w /om/user/ysa/bild/data/second_level_TD_KidsAdults_task1/working_dir_1 -p 'SLURM' --plugin_args "dict(sbatch_args='-N1 -c2 --mem=10G')" --sleep 2 -f ols



#used for kids and adults task2
python /om/user/ysa/l2_mult_regres.py -o /om/user/ysa/bild/data/second_level_TD_KidsAdults_task2/output_dir -t 2 -d /om/user/ysa/bild/data -l1 /om/user/ysa/bild/data/first_level/output_dir -w /om/user/ysa/bild/data/second_level_TD_KidsAdults_task2/working_dir_2 -p 'SLURM' --plugin_args "dict(sbatch_args='-N1 -c2 --mem=10G')" --sleep 2 -f ols
