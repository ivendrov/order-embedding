# patches bug in denotation graph

f = open('node.idx', 'rb')
lines = f.readlines()
f.close()

# put "139222   ride small boat" at the very end"
idx = 139222
badline = lines[idx]
lines = lines[:idx] + lines[idx+1:]
new_idx = len(lines)
print("Replacing:")
print(badline)
print("with:")
badline = badline.replace("139222", str(new_idx))
print(badline)
lines.append(badline)
with open('node.idx', 'wb') as f:
    f.writelines(lines)


# replace in train and the global node-tree.txt (neither 139222 occurs in dev and test)
splits = ['', 'train/']
for split in splits:
    filename = split + 'node-tree.txt'
    f = open(filename, 'rb')
    lines = f.readlines()
    f.close()
    for i in range(len(lines)):
        tokens = lines[i].strip().split('\t')
        if int(tokens[0]) == idx or int(tokens[2]) == idx:
            caps = set(tokens[3:])
            if "998845445.jpg#3" in caps:
                # replace
                if int(tokens[0]) == idx:
                    tokens[0] = str(new_idx)
                if int(tokens[2]) == idx:
                    tokens[2] = str(new_idx)

                new_line = '\t'.join(tokens) + '\n'
                print("Replacing")
                print(lines[i])
                print("with")
                print(new_line)
                lines[i] = new_line

    with open(filename, 'wb') as f:
        f.writelines(lines)













