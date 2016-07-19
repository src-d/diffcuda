import os
from shutil import copyfile
import sys
from subprocess import call

from unidiff import PatchSet


def main():
    base = sys.argv[1]
    patches = sys.argv[2]
    output = sys.argv[3]
    counter = 1
    index = []
    files = os.listdir(patches)
    pc = 0
    for pfile in sorted(files, reverse=True):
        sys.stdout.write("\r%d" % pc)
        pc += 1
        pfile = os.path.join(patches, pfile)
        try:
            patch = PatchSet.from_filename(pfile)
        except:
            continue
        base_counter = counter
        for f in patch.modified_files:
            fn = "%05d_after_%s" % (counter, f.path.replace("/", "_"))
            srcf = os.path.join(base, f.path)
            if os.path.exists(srcf):
                index.append([fn])
                copyfile(srcf, os.path.join(output, fn))
                counter += 1
        call(["git", "apply", "-R", "--reject", pfile], cwd=base)
        counter = base_counter
        for f in patch.modified_files:
            fn = "%05d_before_%s" % (counter, f.path.replace("/", "_"))
            srcf = os.path.join(base, f.path)
            if os.path.exists(srcf):
                index[counter - 1].insert(0, fn)
                copyfile(srcf, os.path.join(output, fn))
                counter += 1
    with open(os.path.join(output, "index.txt"), "w") as fout:
        for p in index:
            fout.write("%s %s\n" % tuple(p))
    print()

if __name__ == "__main__":
    sys.exit(main())
