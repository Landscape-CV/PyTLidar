# Optimum Metrics

These are all of the options for optimum metrics available through the GUI or CLI

---

## Metrics

### Mean Distance Metrics
- `all_mean_dis`: Mean distance for all cylinders.
- `trunk_mean_dis`: Mean distance for trunk cylinders.
- `branch_mean_dis`: Mean distance for branch cylinders.
- `1branch_mean_dis`: Mean distance for first-order branch cylinders.
- `2branch_mean_dis`: Mean distance for second-order branch cylinders.
- `trunk+branch_mean_dis`: Combined mean distance for trunk and branch cylinders.
- `trunk+1branch_mean_dis`: Combined mean distance for trunk and first-order branch cylinders.
- `trunk+1branch+2branch_mean_dis`: Combined mean distance for trunk, first-order, and second-order branch cylinders.
- `1branch+2branch_mean_dis`: Combined mean distance for first-order and second-order branch cylinders.

### Maximum Distance Metrics
- `all_max_dis`: Maximum distance for all cylinders.
- `trunk_max_dis`: Maximum distance for trunk cylinders.
- `branch_max_dis`: Maximum distance for branch cylinders.
- `1branch_max_dis`: Maximum distance for first-order branch cylinders.
- `2branch_max_dis`: Maximum distance for second-order branch cylinders.
- `trunk+branch_max_dis`: Combined maximum distance for trunk and branch cylinders.
- `trunk+1branch_max_dis`: Combined maximum distance for trunk and first-order branch cylinders.
- `trunk+1branch+2branch_max_dis`: Combined maximum distance for trunk, first-order, and second-order branch cylinders.
- `1branch+2branch_max_dis`: Combined maximum distance for first-order and second-order branch cylinders.

### Mean Plus Maximum Distance Metrics
- `all_mean+max_dis`: Combined mean and maximum distance for all cylinders.
- `trunk_mean+max_dis`: Combined mean and maximum distance for trunk cylinders.
- `branch_mean+max_dis`: Combined mean and maximum distance for branch cylinders.
- `1branch_mean+max_dis`: Combined mean and maximum distance for first-order branch cylinders.
- `2branch_mean+max_dis`: Combined mean and maximum distance for second-order branch cylinders.
- `trunk+branch_mean+max_dis`: Combined mean and maximum distance for trunk and branch cylinders.
- `trunk+1branch_mean+max_dis`: Combined mean and maximum distance for trunk and first-order branch cylinders.
- `trunk+1branch+2branch_mean+max_dis`: Combined mean and maximum distance for trunk, first-order, and second-order branch cylinders.
- `1branch+2branch_mean+max_dis`: Combined mean and maximum distance for first-order and second-order branch cylinders.

### Standard Deviation Metrics
- `tot_vol_std`: Standard deviation of total volume.
- `trunk_vol_std`: Standard deviation of trunk volume.
- `branch_vol_std`: Standard deviation of branch volume.
- `trunk+branch_vol_std`: Combined standard deviation of trunk and branch volume.
- `tot_are_std`: Standard deviation of total surface area.
- `trunk_are_std`: Standard deviation of trunk surface area.
- `branch_are_std`: Standard deviation of branch surface area.
- `trunk+branch_are_std`: Combined standard deviation of trunk and branch surface area.
- `trunk_len_std`: Standard deviation of trunk length.
- `branch_len_std`: Standard deviation of branch length.
- `trunk+branch_len_std`: Combined standard deviation of trunk and branch length.
- `branch_num_std`: Standard deviation of branch count.

### Branch Order Distribution Metrics
- `branch_vol_ord3_mean`: Mean volume for branches of order 3.
- `branch_are_ord3_mean`: Mean surface area for branches of order 3.
- `branch_len_ord3_mean`: Mean length for branches of order 3.
- `branch_num_ord3_mean`: Mean branch count for branches of order 3.
- `branch_vol_ord3_max`: Maximum volume for branches of order 3.
- `branch_are_ord3_max`: Maximum surface area for branches of order 3.
- `branch_len_ord3_max`: Maximum length for branches of order 3.
- `branch_num_ord3_max`: Maximum branch count for branches of order 3.
- `branch_vol_ord6_mean`: Mean volume for branches of order 6.
- `branch_are_ord6_mean`: Mean surface area for branches of order 6.
- `branch_len_ord6_mean`: Mean length for branches of order 6.
- `branch_num_ord6_mean`: Mean branch count for branches of order 6.
- `branch_vol_ord6_max`: Maximum volume for branches of order 6.
- `branch_are_ord6_max`: Maximum surface area for branches of order 6.
- `branch_len_ord6_max`: Maximum length for branches of order 6.
- `branch_num_ord6_max`: Maximum branch count for branches of order 6.

### Cylinder Distribution Metrics
- `cyl_vol_dia10_mean`: Mean volume for cylinders with diameter ≤ 10.
- `cyl_are_dia10_mean`: Mean surface area for cylinders with diameter ≤ 10.
- `cyl_len_dia10_mean`: Mean length for cylinders with diameter ≤ 10.
- `cyl_vol_dia10_max`: Maximum volume for cylinders with diameter ≤ 10.
- `cyl_are_dia10_max`: Maximum surface area for cylinders with diameter ≤ 10.
- `cyl_len_dia10_max`: Maximum length for cylinders with diameter ≤ 10.
- `cyl_vol_dia20_mean`: Mean volume for cylinders with diameter ≤ 20.
- `cyl_are_dia20_mean`: Mean surface area for cylinders with diameter ≤ 20.
- `cyl_len_dia20_mean`: Mean length for cylinders with diameter ≤ 20.
- `cyl_vol_dia20_max`: Maximum volume for cylinders with diameter ≤ 20.
- `cyl_are_dia20_max`: Maximum surface area for cylinders with diameter ≤ 20.
- `cyl_len_dia20_max`: Maximum length for cylinders with diameter ≤ 20.

### Cylinder Zenith Distribution Metrics
- `cyl_vol_zen_mean`: Mean volume for cylinders based on zenith angle.
- `cyl_are_zen_mean`: Mean surface area for cylinders based on zenith angle.
- `cyl_len_zen_mean`: Mean length for cylinders based on zenith angle.
- `cyl_vol_zen_max`: Maximum volume for cylinders based on zenith angle.
- `cyl_are_zen_max`: Maximum surface area for cylinders based on zenith angle.
- `cyl_len_zen_max`: Maximum length for cylinders based on zenith angle.

### Surface Coverage Metrics
- `all_mean_surf`: Mean surface coverage for all cylinders.
- `trunk_mean_surf`: Mean surface coverage for trunk cylinders.
- `branch_mean_surf`: Mean surface coverage for branch cylinders.
- `1branch_mean_surf`: Mean surface coverage for first-order branch cylinders.
- `2branch_mean_surf`: Mean surface coverage for second-order branch cylinders.
- `trunk+branch_mean_surf`: Combined mean surface coverage for trunk and branch cylinders.
- `trunk+1branch_mean_surf`: Combined mean surface coverage for trunk and first-order branch cylinders.
- `trunk+1branch+2branch_mean_surf`: Combined mean surface coverage for trunk, first-order, and second-order branch cylinders.
- `1branch+2branch_mean_surf`: Combined mean surface coverage for first-order and second-order branch cylinders.
- `all_min_surf`: Minimum surface coverage for all cylinders.
- `trunk_min_surf`: Minimum surface coverage for trunk cylinders.
- `branch_min_surf`: Minimum surface coverage for branch cylinders.
- `1branch_min_surf`: Minimum surface coverage for first-order branch cylinders.
- `2branch_min_surf`: Minimum surface coverage for second-order branch cylinders.
- `trunk+branch_min_surf`: Combined minimum surface coverage for trunk and branch cylinders.
- `trunk+1branch_min_surf`: Combined minimum surface coverage for trunk and first-order branch cylinders.
- `trunk+1branch+2branch_min_surf`: Combined minimum surface coverage for trunk, first-order, and second-order branch cylinders.
- `1branch+2branch_min_surf`: Combined minimum surface coverage for first-order and second-order branch cylinders.

---
