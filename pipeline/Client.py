# -*- coding: utf-8 -*-
"""
Created on Fri Dec 2 12:35:03 2016

@author: Federico A. Garcia Calabria
"""

#Imports
import socket
import sys
import struct
import re
import argparse

from classifier import Classifier

clf = Classifier("./")

# Read message length and unpack it into an integer
def recv_msg(sock):
    raw_msglen = recvall(sock, 4)

    if not raw_msglen:
        return None

    msglen = struct.unpack('>I', raw_msglen)[0]

    return recvall(sock, msglen)

# Helper function to recv n bytes or return None if EOF is hit
def recvall(sock, n):
    data = b''

    while len(data) < n:
        packet = sock.recv(n - len(data))

        if not packet:
            return None

        data = data + packet

    return data

#Main function
def main():
    #Arguments
    parser = argparse.ArgumentParser(description='WSDM Cup Client')
    parser.add_argument('-d', action = "store", dest = "d", help = 'HOST_NAME:PORT', required = True)
    parser.add_argument('-a', action = "store", dest="a", help = 'AUTHENTICATION_TOKEN', required = True)
    args = parser.parse_args()

    #Variable Definition
    host = args.d[0:args.d.find(":")]
    port = int(args.d[args.d.find(":")+1:])

    auth_token = args.a.encode() + b'\r\n' # for the server to start sending revisions

    metadata = b'' # for metadata
    metadataHeader = '' #for metadataHeader

    revision = b'' # for revision
    revisionHeader = '' #for revisionHeader

    counter = 1 # for demultiplexer
    flagRevisionHeader = False # for revisionHeader
    flagMetadataHeader = False # for metadataHeader

    revisionID = '' # output revisionID
    vandalismScore = '' #output vandalismScore

    answers = 'REVISION_ID,VANDALISM_SCORE\r\n' # CVS Header and upcoming results

    #Creates a AF_INET, STREAM socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    except socket.error:
        print('ERROR: Cannot create socket')

    print('IN PROGRESS: Socket Created')

    # Connect to remote server
    s.connect((host, port))
    print('IN PROGRESS: Connection Established')

    # Send some data to remote server
    try:
        s.sendall(auth_token)
        print('IN PROGRESS: Auth token sent')
    except socket.error:
        # Send failed
        print('ERROR: Cannot send message to server')
        sys.exit()

    #Communicate with the server
    while True:
        #Get message
        reply = recv_msg(s)

        #Keep the Headers in case they are necessary
        if flagMetadataHeader == False and counter % 2 != 0:
            searchOut = re.search(r'[0-9]+', reply.decode('utf-8'))
            metadataHeader = reply.decode('utf-8')[0:searchOut.span()[0]]
            flagMetadataHeader = True

        if flagRevisionHeader == False and reply.decode('utf-8').find("<page>") != -1 and counter % 2 == 0:
            length = reply.decode('utf-8').find("<page>")
            revisionHeader = reply.decode('utf-8')[0:length]
            flagRevisionHeader = True

        #Demultiplex metadata and revision. Call to classifier. Re-initialize revision and metadata. Answer Server
        if counter % 2 == 0 and reply != None:
            revision = revision + reply

            revisionID, vandalismScore = clf.predict_proba(metadata.decode('utf-8'), revision.decode('utf-8'))

            revision = b''
            metadata = b''

            answers = answers + revisionID + ',' + vandalismScore + '\r\n'

            try:
                s.sendall(answers.encode())
                answers = ''

            except socket.error:
                # Send failed
                print('ERROR: Cannot send message to server')
                sys.exit()

        elif counter % 2 != 0 and reply != None:
            metadata = metadata + reply

        #No more data coming from the Server
        if reply == None:
            print('IN PROGRESS: Revisions processed completely')
            break;

        counter = counter + 1

    #Closes socket
    s.close()
    print('COMPLETE')

#Program Execution
if __name__ == main():
    main()

