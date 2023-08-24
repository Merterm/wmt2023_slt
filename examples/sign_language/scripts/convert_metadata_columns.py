import re

def main():
    with open("/afs/cs.pitt.edu/usr0/mei13/research/sign/slt_how2sign_wicv2023/wmt_slt23/metadata_train.csv", "r") as f, open("/afs/cs.pitt.edu/usr0/mei13/research/sign/slt_how2sign_wicv2023/wmt_slt23/signsuisse_metadata.tsv", "w") as f_w:
        f.readline()
        f_w.write('id\tsigns_file\tsigns_offset\tsigns_length\tsigns_type\tsigns_lang\ttranslation\ttranslation_lang\tglosses\ttopic\tsigner_id\n')
        lines = f.readlines()
        for raw_line in lines:
            line = re.split(r',\s*(?=(?:[^"]*"[^"]*")*[^"]*$)', raw_line)
            # print(line)
            if ("de" in line[2]) and ("dsgs" in line[3]):
                f_w.write(f'{line[0]}\t{line[0]}.pose\t0\t{line[9]}\tmediapipe\tdsgs\t{line[1]}\tde\t{line[1]}\tNone\tNone\n')


if __name__ == "__main__":
    main()