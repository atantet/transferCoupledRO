INCDIR=-I$(HOME)/.local/include/ -I/opt/local/include/
LIBDIR=-L$(HOME)/.local/lib/ -L/opt/local/lib/
LIBS=-lgsl -lconfig++ -lergopack -lgomp
CONFIG=../cfg/readConfig

CC=g++ -fopenmp
WARN=-Wall -Wformat=0
CFLAGS=$(WARN) -pedantic-errors -O3
LDFLAGSSTD=-undefined dynamic_lookup
LDFLAGSGDB=$(LDFLAGSSTD) -gdb

LDFLAGS=$(LDFLAGSSTD)
#LDFLAGS=$(LDFLAGSGDB)

all:$(EXE).out
	mkdir -p ../results/transfer/forwardTransition
	mkdir -p ../results/transfer/backwardTransition
	mkdir -p ../results/transfer/initDist
	mkdir -p ../results/transfer/finalDist
	mkdir -p ../results/transfer/gridMem
	mkdir -p ../results/transfer/mask
	mkdir -p ../results/grid/

$(EXE).o:$(EXE).cpp
	$(CC) -c $(CFLAGS) $(INCDIR) -o $@ $<

$(CONFIG).o:$(CONFIG).cpp
	$(CC) -c $(CFLAGS) $(INCDIR) -o $@ $<

$(EXE).out:$(EXE).o $(CONFIG).o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBDIR) $(LIBS)

clean:
	rm -f $(EXE).o $(EXE).out

cleanall:
	rm -f $(EXE).o $(CONFIG).o $(EXE).out
