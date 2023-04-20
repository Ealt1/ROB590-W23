% Number of robots
N = 35;
% Workspace
numRows = 30;
numCols = 30;
parameters = [...
 29,16,5,10,0.506699;
 4,22,2,11,0.7378;
 27,2,17,13,0.583346;
 26,11,14,6,0.548082;
 24,8,9,13,0.545961;
 27,24,14,19,0.937376;
 7,28,4,4,0.674291;
 26,26,7,12,0.781564;
 20,25,2,4,0.601093;
 19,12,16,22,0.682119;
 21,15,23,19,0.67925;
 25,1,21,0,0.639882;
 19,11,26,4,0.634068;
 23,13,13,28,0.576693;
 21,4,21,29,0.562853;
 8,21,29,14,0.643437;
 0,4,16,14,0.590884;
 13,17,12,12,0.511841;
 23,28,5,20,0.71662;
 13,28,2,18,0.676595;
 25,26,22,17,0.891903;
 10,8,6,6,0.756035;
 5,23,3,7,0.996521;
 21,24,24,29,0.937269;
 2,26,21,9,0.895749;
 14,3,17,3,0.626148;
 5,5,16,19,0.523377;
 26,14,28,22,0.598346;
 25,27,17,14,0.727546;
 20,10,27,14,0.565477;
 4,13,29,11,0.761559;
 20,24,1,1,0.634205;
 19,27,10,21,0.611927;
 21,1,6,29,0.706977;
 24,21,16,9,0.641575;];

% Paths
Paths = cell(35,1);
Paths{1} = [977 945 913 881 849 817 785 753 721 721 721 689 657 625 593 561 560 559 527 526 494 462 461 460 428 396 395 363 331 299 267 235 203];
Paths{2} = [183 182 181 180 179 178 177 176 175 174 173 141 109 108];
Paths{3} = [899 867 868 869 870 871 872 873 841 809 777 745 713 681 649 617 618 619 620 621 622 590];
Paths{4} = [876 844 843 842 841 840 808 776 744 712 680 648 616 584 552 520 519 487];
Paths{5} = [809 810 778 746 714 682 650 618 619 587 555 556 524 492 460 428 396 397 365 333 334];
Paths{6} = [921 889 890 891 859 860 828 796 764 732 700 668 667 666 665 664 663 662 630 598 566 534 533 532 500];
Paths{7} = [285 284 283 282 281 280 279 278 277 276 275 274 273 272 271 270 269 268 267 266 265 264 263 262 230 198 166 165];
Paths{8} = [891 892 860 828 796 764 763 762 730 729 728 696 695 694 693 661 629 597 565 564 563 531 499 498 466 434 402 401 400 399 367 335 303 271 270 269];
Paths{9} = [698 666 634 602 570 569 537 505 504 503 502 470 469 468 436 404 372 340 339 338 306 305 304 272 240 239 238 237 236 204 172 171 139 107 106 105 104 103 102 101];
Paths{10} = [653 621 622 623 624 625 626 627 628 629 630 598 566 567];
% Paths{11} = [720 721 753 785 786 787 788];
Paths{11} = [720 719 720 721 753 785 786 787 788];
Paths{12} = [834 802 801 769 737 705];
Paths{13} = [652 684 683 682 681 680 679 711 743 775 807 806 838 870 869];
Paths{14} = [782 750 751 752 720 688 656 624 625 626 627 628 629 630 631 632 633 634 635 603 604 572 540 541 509 477];
Paths{15} = [709 741 742 743 744 745 746 747 748 749 750 751 752 753 785 786 787 755 756 757 758 759 760 728 729 730 731 732 733 734];
Paths{16} = [310 309 308 307 306 338 370 402 434 466 498 530 562 594 626 658 657 656 688 720 752 784 816 848 880 912 944 943 975];
Paths{17} = [37 38 39 40 41 73 105 137 138 170 202 234 266 267 299 331 332 333 334 366 398 430 431 432 464 496 495 527 559];
Paths{18} = [466 434 433 432 431 430 429];
Paths{19} = [797 765 733 732 731 730 729 697 665 633 601 569 537 505 504 503 471 439 407 406 405 373 341 309 277 245 213];
Paths{20} = [477 445 413 412 411 410 378 346 345 344 343 342 341 340 339 307 275 243 211 179 147 115];
Paths{21} = [859 858 857 856 855 854 853 852 851 850 818 786 754];
Paths{22} = [361 329 297 265 264 263 231];
Paths{23} = [216 248 247 246 214 213 212 211 210 209 208 207 206 205 204 172 140 139 138 137 136];
Paths{24} = [729 730 731 763 795 827 859 860 861 862 830];
Paths{25} = [123 155 187 219 251 283 315 347 379 411 412 413 445 477 509 541 573 605 637 669 668 667 666 665 664 663 662 661 660 659 658 657 656 655 654 653 652 684 683 682 714];
Paths{26} = [484 516 548 580];
Paths{27} = [198 230 262 294 326 327 328 360 392 393 394 395 396 397 398 430 431 432 464 496 497 498 530 562 563 564];
Paths{28} = [879 880 912 944 945 946 947 948 949 950 951];
Paths{29} = [860 828 827 826 825 824 824 823 822 821 820 819 787 755 723 691 690 658 658 626 625 624 623 591];
Paths{30} = [683 715 747 779 811 843 875 907 908 909 910 911];
Paths{31} = [174 206 238 270 271 303 335 334 366 398 430 462 494 526 527 559 591 623 622 654 686 718 750 749 781 813 845 877 909 941 973 972];
Paths{32} = [697 665 633 601 600 568 567 535 534 533 532 500 499 467 435 403 371 370 369 337 336 335 303 271 270 269 268 267 266 265 264 263 262 230 229 228 227 226 194 162 130 98 66];
Paths{33} = [668 667 635 603 571 539 538 538 537 505 473 472 471 439 407 406 374];
Paths{34} = [706 707 708 709 710 711 712 713 714 715 716 684 685 686 687 687 688 689 690 691 692 693 694 695 696 697 665 633 601 569 537 505 473 474 475 443 411 379 347 315 283 284 285 286 254];
Paths{35} = [822 821 820 788 756 755 723 691 690 689 688 687 655 623 591 590 589 557 556 555 554];

for n = 1:35
	y =  floor(Paths{n}/32) - 1;
	x = mod(Paths{n},32) -1;
	Paths{n} = y + x*30 +1;
    p = 2;
%     while p <= length(Paths{n})
%         if Paths{n}(p) == Paths{n}(p-1)
%             Paths{n}(p) = [];
%         else
%             p = p + 1;
%         end
%     end
end

% Init/Final Conditions, Delay Prob
initial_locations = zeros(1,N);
final_locations = zeros(1,N);
for i = 1:N
   initial_locations(i) = Paths{i}(1);
   final_locations(i) = Paths{i}(end);
end
% Obstacles
obstacles = find([;
0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,;
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,;
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,;
0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,;
0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,;
0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,;
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,1,0,;
0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,;
0,0,0,0,1,0,0,1,0,1,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,;
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,;
0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,;
0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,;
0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,;
0,0,1,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,;
1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,;
0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,;
0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,;
0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1,0,0,1,0,0,0,0,0,0,0,;
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,;
1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,;
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,;
0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,;
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,;
0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,;
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,;
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,;
1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,;
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,;
1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,;
0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,;
]);
ws = create_workspace(numRows, numCols, obstacles);