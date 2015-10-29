# patches bug in denotation graph

f = open('node.idx', 'rb')
lines = f.readlines()
f.close()

# fix node.idx by putting "139222   ride small boat" at the very end
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

def tok2line(toks):
    return '\t'.join(toks) + '\n'

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

                new_line = tok2line(tokens)
                print("Replacing")
                print(lines[i])
                print("with")
                print(new_line)
                lines[i] = new_line

    with open(filename, 'wb') as f:
        f.writelines(lines)


# replace in node-cap.map
f = open('node-cap.map', 'rb')
lines = f.readlines()
f.close()

badline = lines[idx]
tokens = badline.strip().split('\t')
bad_capid = tokens[-1]
lines[idx] = tok2line(tokens[:-1])
lines.append(tok2line([str(len(lines)), bad_capid]))

with open('node-cap.map', 'wb') as f:
    f.writelines(lines)











