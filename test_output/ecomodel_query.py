import csv
import utm
import numpy as np
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            prog="Ecomodel Query",
            description="Determine local environment branch characteristics.",
        )

    parser.add_argument('input_file', help="Input CSV file with lizard locations.")
    parser.add_argument('cylinder_file', help="Cylinder txt file output from Ecomodel.")
    parser.add_argument('output_file', help="Output CSV file with lizard locations AND branch length metrics.")
    parser.add_argument('-v', '--voxel_size', help="Size of box in meters around lizard location to include cylinder lengths from.")

    args = parser.parse_args()

    cylinder_data = np.loadtxt(args.cylinder_file)

    if not args.voxel_size:
        voxel_size = 2
    else:
        voxel_size = float(args.voxel_size) # meters


    write_data = [["alpha_tag", "date", "time", "perch_height", "perch_width", "perch_type", "Notes", 
                   "Longitude", "Latitude", "source_file", "fixed_notes", "species", "sex", "mass_g", 
                   "length_twigs_m", "length_small_branches_m", "length_medium_branches_m", "length_large_branches_m"]]

    with open(args.input_file, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            if row[3] == 'NA' or row[7] == 'NA' or row[8] == 'NA':
                continue
            
            # Assumption is that height normalization will shift points to be their height above ground. 
            height_m = float(row[3]) / 100
            longitude = float(row[7])
            latitude = float(row[8])

            easting, northing, zone_number, zone_letter = utm.from_latlon(latitude, longitude)

            half_voxel = voxel_size / 2
            x = easting
            y = northing
            z = height_m

            cube_min = np.array([x - half_voxel, y - half_voxel, z - half_voxel])
            cube_max = np.array([x + half_voxel, y + half_voxel, z + half_voxel])
            
            voxel_mask = np.all((cylinder_data[:,0:3] >= cube_min) & (cylinder_data[:,0:3] <= cube_max), axis=1)

            voxel_cylinders = cylinder_data[voxel_mask]


            # Constants for branch buckets
            low = 0
            twig_max = 0.02
            small_max = 0.05
            med_max = 0.10


            twig_length = 0
            small_length = 0
            medium_length = 0
            large_length = 0
            for idx in range(voxel_cylinders.shape[0]):
                
                radius = voxel_cylinders[idx, 3]
                actual_length = voxel_cylinders[idx, 7]

                if low < radius <= twig_max:
                    twig_length+= actual_length
                elif twig_max < radius <= small_max:
                    small_length+= actual_length
                elif small_max < radius <= med_max:
                    medium_length+= actual_length
                else:
                    large_length+= actual_length

            write_data.append(row + [twig_length, small_length, medium_length, large_length])


    with open(args.output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for data in write_data:
            writer.writerow(data)

    