#!/usr/bin/env bash

#
# Goodness-of-fit tests in CR regions
# (interpreted as DNN bins with a S/sqrt(B) ratio < 0.05)
#

# Default values for arguments
unblind=false
masses="350 700 1000 2000"
input_dir="/data/dust/user/jwulff/inference/stat_tests/cards/orig"
output_dir="/data/dust/user/jwulff/inference/stat_tests"
other_masses="250 260 270 280 300 320 400 450 500 550 600 650 750 800 850 900 1250 1500 1750 2500 3000"

# Parse CLI arguments
while getopts "u:m:i:o:" flag; do
    case "${flag}" in
        u) unblind=true ;;
        m) masses=${OPTARG} ;; # Accept space-separated masses directly
        i) input_dir=${OPTARG} ;;
        o) output_dir=${OPTARG} ;;
        *) echo "Usage: $0 [-u (unblind)] [-m masses \"(space-separated)\"] [-i input_dir] [-o output_dir]"; exit 1 ;;
    esac
done

echo "Running with the following parameters:"
echo "Unblind: ${unblind}"
echo "Masses: ${masses}"
echo "Input directory: ${input_dir}"
echo "Output directory: ${output_dir}"

if [ "${unblind}" = true ]; then
    output_dir="${output_dir}/unblinded"
else
    output_dir="${output_dir}/blinded"
fi
# Create output directory if it doesn't exist
if [ ! -d "${output_dir}" ]; then
    echo "Creating output directory ${output_dir}..."
    mkdir -p "${output_dir}"
fi

remove_sensitive_bins() {
    local src_card="$1"
    local dst_dir="$2"

    # remove bins with a signal-to-noise ratio > 0.05
    remove_shape_bins.py "$1" -d "$2" '*,STN>0.05'

    # remove the datacard altogether if no bins are left
    local name="$( basename ${src_card} )"
    local dst_card="${dst_dir}/${name}"
    if [ ! -z "$( cat "${dst_card}" | grep -Po "observation\s+0\.0$" )" ]; then
        name="${name:9:-4}"
        echo -e "removing empty datacard and shape \x1b[0;49;33m${name}\x1b[0m"
        (cd "${dst_dir}" && rm "datacard_${name}.txt" "shapes_${name}.root" )
    fi
}

gof_tests() {
    local result_dir="$1"
    local cards_dir="$2"
    local spin="$3"
    local mass="$4"
    if [ -z "${mass}" ]; then
        >&2 echo "Usage: gof_tests <result_dir> <cards_dir> <spin> <mass>"
        return "1"
    fi

    # create CR cards first if not existing
    local result_cards_dir="${result_dir}/cards"
    local dc="${result_cards_dir}/datacard.txt"
    if [ ! -f "${dc}" ]; then
        echo "Creating dir ${result_cards_dir}... for the datacards that will be used"
        mkdir -p "${result_cards_dir}"

        if [ "${unblind}" = false ]; then
            # update cards
            for y in 2016APV 2016 2017 2018; do
                for ch in etau mutau tautau; do
                    for cat in resolved1b_noak8 resolved2b_first boosted_notres2b; do
                        local src_card="${cards_dir}/datacard_cat_${y}_${ch}_${cat}_os_iso_spin_${spin}_mass_${mass}.txt"
                        if [ ! -f "${src_card}" ]; then
                            >&2 echo -e "File not found: \x1b[0;49;31m${src_card}\x1b[0m"
                            continue
                        fi
                        remove_sensitive_bins "${src_card}" "${result_cards_dir}" &
                    done
                done
                # parallelize over channel and category
                wait
            done
        else
            # Copy all datacards directly if unblind is true 
            for y in 2016APV 2016 2017 2018; do
                for ch in etau mutau tautau; do
                    for cat in resolved1b_noak8 resolved2b_first boosted_notres2b; do
                        local src_card="${cards_dir}/datacard_cat_${y}_${ch}_${cat}_os_iso_spin_${spin}_mass_${mass}.txt"
                        if [ ! -f "${src_card}" ]; then
                            >&2 echo -e "File not found: \x1b[0;49;31m${src_card}\x1b[0m"
                            continue
                        fi
                        # check if corresponding shape file exists (replace datacard name with shape name and .txt with .root)
                        local src_shape="${src_card/datacard_/shapes_}"
                        src_shape="${src_shape/.txt/.root}"
                        if [ ! -f "${src_shape}" ]; then
                            >&2 echo -e "File not found: \x1b[0;49;31m${src_shape}\x1b[0m"
                            continue
                        fi
                        local dst_card="${result_cards_dir}/datacard_cat_${y}_${ch}_${cat}_os_iso_spin_${spin}_mass_${mass}.txt"
                        local dst_shape="${result_cards_dir}/shapes_cat_${y}_${ch}_${cat}_os_iso_spin_${spin}_mass_${mass}.root"
                        # copy datacard and shape file
                        cp "${src_card}" "${dst_card}"
                        cp "${src_shape}" "${dst_shape}"
                    done
                done
                # parallelize over channel and category
                wait
            done
        fi

        # combine datacards
        (
            cd "${result_cards_dir}" &&
            combineCards.py datacard_*.txt > "${dc}"
        ) || return "$?"
    fi

    # create workspace
    local ws="${result_dir}/workspace.root"
    if [ ! -f "${ws}" ]; then
        cd "${result_cards_dir}"
        text2workspace.py "${dc}" --out "${ws}" || return "$?"
    fi

    # observed GOF
    local gof_obs="${result_dir}/higgsCombineTest.GoodnessOfFit.mH120.root"
    if [ ! -f "${gof_obs}" ]; then
        (
            cd "${result_dir}" &&
            combine -M GoodnessOfFit \
                -d "${ws}" \
                --algo saturated \
                $( [ "${unblind}" = false ] && echo "--freezeParameters r --setParameters r=0" )
        ) || return "$?"
    fi

    # toy GOFs
    local toys="500"
    local gof_toys="${result_dir}/higgsCombineTest.GoodnessOfFit.mH120.${toys}toys.root"
    if [ ! -f "${gof_toys}" ]; then
        local cores="25"
        local toys_per_core="$(( toys / cores ))"
        local seed
        (
            cd "${result_dir}" &&
            for seed in $( seq 1 "${cores}" ); do
                combine -M GoodnessOfFit \
                    -d "${ws}" \
                    --algo saturated \
                    $( [ "${unblind}" = false ] && echo "--freezeParameters r --setParameters r=0" ) \
                    --seed "${seed}" \
                    -t "${toys_per_core}" \
                    --toysFrequentist &
            done
            wait &&
            hadd tmp.root higgsCombineTest.GoodnessOfFit.mH120.*.root
            rm higgsCombineTest.GoodnessOfFit.mH120.*.root
            mv tmp.root "${gof_toys}"
        ) || return "$?"
    fi

    # extract results
    local gof_json="${result_dir}/gof.json"
    if [ ! -f "${gof_json}" ]; then
        (
            cd "${result_dir}" &&
            combineTool.py -M CollectGoodnessOfFit \
                --input "${gof_obs}" "${gof_toys}" \
                -o "${gof_json}"
        ) || return "$?"
    fi

    # plot
    (
        cd "${result_dir}" &&
        plotGof.py gof.json \
            --statistic saturated \
            --mass 120.0 \
            -o gof_test \
            --title-right="X \rightarrow HH \rightarrow bb\tau\tau, Run 2"
    )
}

#
# bias tests
#

bias_tests() {
    local result_dir="$1"
    local cards_dir="$2"
    local spin="$3"
    local mass="$4"
    local r="$5"
    if [ -z "${r}" ]; then
        >&2 echo "Usage: gof_tests <result_dir> <cards_dir> <spin> <mass> <r>"
        return "1"
    fi

    local shell_is_zsh=$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    # create a combined datacard first if not existing
    local dc="${result_dir}/datacard.txt"
    if [ ! -f "${dc}" ]; then
        # collect cards
        local cards=""
        for y in 2016APV 2016 2017 2018; do
            for ch in etau mutau tautau; do
                for cat in resolved1b_noak8 resolved2b_first boosted_notres2b; do
                    local src_card="${cards_dir}/datacard_cat_${y}_${ch}_${cat}_os_iso_spin_${spin}_mass_${mass}.txt"
                    if [ ! -f "${src_card}" ]; then
                        >&2 echo -e "File not found: \x1b[0;49;31m${src_card}\x1b[0m"
                        continue
                    fi
                    cards="${cards} ${src_card}"
                done
            done
        done

        # combine them
        mkdir -p "${result_dir}"
        eval "combineCards.py ${cards} > ${dc}" || return "$?"
    fi

    # create workspace
    local ws="${result_dir}/workspace.root"
    if [ ! -f "${ws}" ]; then
        text2workspace.py "${dc}" --out "${ws}" || return "$?"
    fi

    # generate toys
    local toys="560"
    local cores="35"
    local toys_per_core="$(( toys / cores ))"
    toy_file() { local seed="$1"; echo "${result_dir}/higgsCombineTest.GenerateOnly.mH120.${seed}.root"; };
    local seed
    (
        cd "${result_dir}"
        for seed in $( seq 1 "${cores}" ); do
            if [ ! -f "$( toy_file "${seed}" )" ]; then
                combine -M GenerateOnly \
                    -d "${ws}" \
                    --expectSignal "${r}" \
                    --saveToys \
                    --toysFrequentist \
                    --bypassFrequentistFit \
                    --seed "${seed}" \
                    -t "${toys_per_core}" &
            fi
        done
        wait
    )

    # perform fit and print best fitting r
    local r_min="-2"
    local r_max="4"
    fit_file() { local seed="$1"; echo "${result_dir}/higgsCombineTest.MultiDimFit.mH120.${seed}.root"; };
    (
        cd "${result_dir}"
        for seed in $( seq 1 "${cores}" ); do
            if [ ! -f "$( fit_file "${seed}" )" ]; then
                combine -M MultiDimFit \
                    -d "${ws}" \
                    --algo singles \
                    --rMin "${r_min}" \
                    --rMax "${r_max}" \
                    -t "${toys_per_core}" \
                    --toysFrequentist \
                    --toysFile "$( toy_file "${seed}" )" \
                    --seed "${seed}" &
            fi
        done
        wait
    )

    # plot
    (
        cd "${result_dir}" &&
        python3 "${this_dir}/plot_bias.py" \
            "${r}" \
            "higgsCombineTest.MultiDimFit.mH120.*.root" \
            "${r_min}" \
            "${r_max}"
    ) || return "$?"
}

gather_results() {
    local source_dir="$1"
    local target_dir="$2"

    echo "Gathering results from ${source_dir} to ${target_dir}..."
    mkdir -p "${target_dir}"

    # Iterate over all subdirectories in the source directory
    for mass_dir in "${source_dir}"/*; do
        if [ -d "${mass_dir}" ]; then
            mass=$(basename "${mass_dir}")
            # Copy and rename .png and .pdf files
            for file in "${mass_dir}"/*.{png,pdf}; do
                if [ -f "${file}" ]; then
                    extension="${file##*.}"
                    filename="${mass}_$(basename "${file}")"
                    cp "${file}" "${target_dir}/${filename}"
                fi
            done
        fi
    done

    echo "All files have been gathered in ${target_dir}."
}

#
# main entry points
#

# gof tests
for mass in ${masses}; do
    gof_tests \
        "${output_dir}/gof_s0_m${mass}" \
        "${input_dir}" \
        "0" \
        "${mass}"
done

# bias tests
#for mass in ${masses}; do
#    for r in 0 1; do
#        bias_tests \
#            "${output_dir}/bias_s0_m${mass}_r${r}" \
#            "${input_dir}" \
#            "0" \
#            "${mass}" \
#            "${r}"
#    done
#done

# Gather results
#gather_results "${output_dir}" "${output_dir}/collected_results"
