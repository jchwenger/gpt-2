import re
import os
import sys
import time
import regex
import argparse

from six.moves import queue

import gpt_2_simple as gpt2

# disabling some warnings
os.environ["KMP_WARNINGS"] = "off"

def gpt_answer_string(sess, pref, length=250, run=None):
    return gpt2.generate(
        sess,
        run_name=run,
        checkpoint_dir="checkpoint",
        model_name="117M",
        model_dir="models",
        sample_dir="samples",
        return_as_list=True,
        # sample_delim="=" * 20 + "\n",
        prefix=pref,
        # seed=None,
        nsamples=1,
        batch_size=1,
        length=length,
        temperature=1,
        # top_k=0,
        # top_p=5,
        include_prefix=True,
    )[0]

def main(args):

    sess = gpt2.start_tf_sess()
    print(args.run)
    gpt2.load_gpt2(sess, run_name=args.run)

    # hack: dummy generation to get first warnings out of the way
    gpt2.generate(sess, length=1, top_p=5)

    # find first response in gpt stream
    pref_re = regex.compile(
        "(?<=<\|débutderéplique\|>\n).*?(?=\n<\|finderéplique\|>)", regex.DOTALL
    )
    new_pref = ""

    os.system('clear')
    underprint("Le réseau écoute...")

    while True:

        # plain generation, simply get more text
        if args.plain: 

            text = input()
            pref = f"{new_pref+text}"
            end_pref = len(pref)
            l = gpt_answer_string(sess, pref, args.length, run=args.run)

            # prefix riddance
            l_no_pref = l[end_pref:]

            # clean up end of answer
            # https://stackoverflow.com/a/33233868
            try:
                last_ind = re.search("(?s:.*)\.\n", l_no_pref).end()
            except:
                last_ind = len(l_no_pref) - 1
            l_no_pref = l_no_pref[:last_ind-1]
            answer = l_no_pref
            if new_pref == "":
                print("\033[F"+answer)
            else:
                print("\033[F"*2+answer)
            new_pref = pref + l_no_pref

        # chatbot, with one or more answers
        else:

            print()
            text = input("TOI.\n")

            # add end of answer, store length of prefix
            pref = f"{new_pref+text}\n<|finderéplique|>\n"
            end_pref = len(pref)

            l = gpt_answer_string(sess, pref, args.length, run=args.run)

            # prefix riddance
            l_no_pref = l[end_pref:]

            # multiple answers
            if args.choice:

                # regex for all answers
                m = list(regex.finditer(pref_re, l_no_pref))
                print()
                underprint("Toutes les réponses.")
                for i, answer in enumerate(m):
                    print()
                    print(f"{i+1}:\n{answer.group(0)}")
                print()
                choice = input("\tTon choix? ")
                print()
                while not (choice.isdigit() and int(choice) <= len(m)):
                    choice = input(
                        "S'il te plaît, donne-moi le chiffre de ta réponse: "
                    )
                choice = int(choice)
                answer = m[choice - 1].group(0)

                # security: if none, resample
                while not m:
                    l = gpt_answer_string(sess, pref, args.length, run=args.run)
                    l_no_pref = l[end_pref:]
                    m = list(regex.finditer(pref_re, l_no_pref))

                answer_end_ind = m[choice - 1].span()[1]

            # plain chatbot, take the first answer
            else:

                # regex get our first answer
                m = regex.search(pref_re, l_no_pref)

                # security: if none, resample
                while not m:
                    l = gpt_answer_string(sess, pref, args.length)
                    l_no_pref = l[end_pref:]
                    m = regex.search(pref_re, l_no_pref)

                answer = m.group(0)
                answer_end_ind = m.span()[1]

            print()
            print("L'AUTRE.")
            print(f"{answer}")

            new_pref = f"{l[:end_pref+answer_end_ind]}\n<|finderéplique|>\n"

            if args.typewriter:
                with open("typeWriter/data/answer.txt", "w") as o:
                    o.write(answer)

def underprint(msg):
    print(msg)
    print('-'*len(msg))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Termbot with gpt.")

    parser.add_argument(
        "--choice", "-c",
        help="Returns more than one answer that the user can choose from. Default: false",
        action="store_true",
    )

    parser.add_argument(
        "--plain", "-p",
        help="Prints all the output.",
        type=bool,
        default=True
    )

    parser.add_argument(
        "--length", "-l",
        help="The length of the text gpt-2 will generate. Default: 250.",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--run", "-r",
        help="The run to use in checkpoint/",
        type=str,
        default='run1',
    )

    parser.add_argument(
        "--typewriter", "-t",
        help="Write answer to typeWriter/data/answer.txt for Processing",
        action="store_true",
    )

    args = parser.parse_args()

    main(args)
    # [END speech_transcribe_infinite_streaming]
