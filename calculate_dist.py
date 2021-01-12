def distance(p1, p2):    
    return ((float(p1[0]) - float(p2[0])) ** 2 + (float(p1[1]) - float(p2[1])) ** 2 + (float(p1[2]) - float(p2[2])) ** 2) ** 0.5

def print_distance(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        pc = []
        for line in lines:
            point = line.split(',')
            pc.append(point)


        sum_d = 0
        for i in range(len(pc)):
            base_point = pc[i]
            min_d = 100
            for j in range(len(pc)):
                if i == j: continue
                point = pc[j]
                d = distance(base_point, point)
                if d < min_d:
                    min_d = d
            sum_d += min_d
        
        print("-------------------------------------")
        print("Input file name:", filename)
        print("Distance is ", sum_d)

print_distance('FPS_result.csv')
print_distance('FPS_result_light.csv')
print_distance('FPS_result_random.csv')
