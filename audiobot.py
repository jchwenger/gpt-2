#!/usr/bin/env python

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech API sample application using the streaming API.

NOTE: This module requires the additional dependency `pyaudio`. To install
using pip:

    pip install pyaudio

Example usage:
    python transcribe_streaming_indefinite.py
"""

# [START speech_transcribe_infinite_streaming]
from __future__ import division

import re
import os
import sys
import time
import regex
import argparse

# disabling some warnings
os.environ["KMP_WARNINGS"] = "off"

from google.cloud import speech

import pyaudio
from six.moves import queue

import gpt_2_simple as gpt2

# Audio recording parameters
STREAMING_LIMIT = 290000
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms


def get_current_time():
    return int(round(time.time() * 1000))


def duration_to_secs(duration):
    return duration.seconds + (duration.nanos / float(1e9))


class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk_size):
        self._rate = rate
        self._chunk_size = chunk_size
        self._num_channels = 1
        self._max_replay_secs = 5

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()

        # 2 bytes in 16 bit samples
        self._bytes_per_sample = 2 * self._num_channels
        self._bytes_per_second = self._rate * self._bytes_per_sample

        self._bytes_per_chunk = self._chunk_size * self._bytes_per_sample
        self._chunks_per_second = self._bytes_per_second // self._bytes_per_chunk

    def __enter__(self):
        self.closed = False

        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, *args, **kwargs):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            if get_current_time() - self.start_time > STREAMING_LIMIT:
                self.start_time = get_current_time()
                break
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)


def listen_print_loop(responses, stream):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    responses = (r for r in responses if (r.results and r.results[0].alternatives))

    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        top_alternative = result.alternatives[0]
        transcript = top_alternative.transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()

            num_chars_printed = len(transcript)
        else:
            print(transcript + overwrite_chars)

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                print("Exiting..")
                stream.closed = True
                break

            num_chars_printed = 0


def listen_loop(responses, stream):
    responses = (r for r in responses if (r.results and r.results[0].alternatives))

    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        top_alternative = result.alternatives[0]
        transcript = top_alternative.transcript

        if result.is_final:
            return transcript


def gpt_answer_string(sess, pref, length=250):
    return gpt2.generate(
        sess,
        run_name="run1",
        checkpoint_dir="checkpoint",
        model_name="117M",
        model_dir="models",
        sample_dir="samples",
        return_as_list=True,
        sample_delim="=" * 20 + "\n",
        prefix=pref,
        # seed=None,
        nsamples=1,
        batch_size=1,
        length=length,
        temperature=0.7,
        top_k=0,
        top_p=5,
        include_prefix=True,
    )[0]


def main(args):

    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name="run1")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(
        "/media/default/linux-data/docs/voix-264921-485ba4bc7629.json"
    )
    client = speech.SpeechClient()
    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="fr-FR",
        enable_word_time_offsets=True,
    )
    streaming_config = speech.types.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)

    # hack: dummy generation to get first warnings out of the way
    gpt2.generate(sess, length=1, top_p=5)

    # find first response in gpt stream
    pref_re = regex.compile(
        "(?<=<\|débutderéplique\|>\n).*?(?=\n<\|finderéplique\|>)", regex.DOTALL
    )
    new_pref = ""

    start = True

    with mic_manager as stream:
        while not stream.closed:
            audio_generator = stream.generator()

            if start:
                msg = "GPT écoute..."
                print()
                print(msg)
                print("-" * len(msg))
                start = False

            requests = (
                speech.types.StreamingRecognizeRequest(audio_content=content)
                for content in audio_generator
            )

            responses = client.streaming_recognize(streaming_config, requests)

            text = listen_loop(responses, stream)
            print()
            print("TOI.")
            print(text)

            # add end of answer, store length of prefix
            pref = f"{new_pref+text}\n<|finderéplique|>\n"
            end_pref = len(pref)

            l = gpt_answer_string(sess, pref, args.length)

            # prefix riddance
            l_no_pref = l[end_pref:]

            if args.choice:
                # regex for all answers
                m = list(regex.finditer(pref_re, l_no_pref))
                print()
                print("Toutes les réponses.")
                for i, answer in enumerate(m):
                    print(f"{i+1}:\n  {answer.group(0)}")
                choice = input("choix: ")
                while not (choice.isdigit() and int(choice) <= len(m)):
                    choice = input(
                        "S'il te plaît, donne-moi le chiffre de ta réponse: "
                    )
                choice = int(choice)
                answer = m[choice - 1].group(0)

                # security: if none, resample
                while not m:
                    l = gpt_answer_string(sess, pref, args.length)
                    l_no_pref = l[end_pref:]
                    m = list(regex.finditer(pref_re, l_no_pref))

                answer_end_ind = m[choice - 1].span()[1]

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
            print("GPT.")
            print(f"{answer}")

            with open("answer.txt", "w") as o:
                o.write(answer)

            new_pref = f"{l[:end_pref+answer_end_ind]}\n<|finderéplique|>\n"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Audiobot with gpt.")

    parser.add_argument(
        "--choice", "-c",
        help="Returns more than one answer that the user can choose from. Default: false",
        action="store_true",
    )

    parser.add_argument(
        "--length", "-l",
        help="The length of the text gpt-2 will generate. Default: 250.",
        type=int,
        default=250,
    )

    args = parser.parse_args()

    main(args)
    # [END speech_transcribe_infinite_streaming]
