using CSV
using DataFrames
using Distributions
using GLM

data_dir = "/Users/yoelsanchezaraujo/Documents/bild_stuff";

if pwd() != data_dir
    cd(data_dir)
end

files = readdir(data_dir);
voxels = CSV.read(files[end], delim=',');
roidata = CSV.read(files[7], delim=',');
behav = CSV.read(files[2], delim="\t");

# add a column header for the first column, it's currently empty
names!(voxels, [:subject_id; names(voxels)[2:end]]);
names!(roidata, [:subject_id; names(roidata)[2:end]]);

# if this is not equal to zero then number of voxels varies between participants for some roi
voxels_aresame = sum(colwise(std, voxels[:, 2:end])) == 0;

cols_to_remove = [
    "WM",
    "Ventricle",
    "Unknown",
    "Vent",
    "CSF",
    "White_matter"
];

function remove_cols(data)
    remove_idx = []
    N, P = size(data)
    for (idx, col) in enumerate(names(voxels))
        col_str = string(col)
        for col_ref in cols_to_remove
            if contains(col_str, col_ref)
                push!(remove_idx, idx)
            end
        end
    end
    return data[:, setdiff(1:P, unique(remove_idx))]
end

voxels = remove_cols(voxels);
roidata = remove_cols(roidata);

# doing some preprocessing of the column names to make them lm friendly
names!(
    voxels, 
    [Symbol(replace(split(string(col_name), ".")[1], "-", "_")) 
     for col_name in names(voxels)]
);

names!(
    roidata,
    [Symbol(replace(split(string(col_name), ".")[1], "-", "_"))
     for col_name in names(roidata)]
);


# checking if it's already ordered by rows
sum(behav[:subject_id] .== voxels[:subject_id]) == size(voxels, 1)

# combine dataframes with behavioral data
voxels_behav = [behav voxels[:, 2:end]];
roidata_behav = [behav roidata[:, 2:end]];

